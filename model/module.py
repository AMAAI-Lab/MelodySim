import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import lightning as L

from transformers import AutoModel, Wav2Vec2FeatureExtractor

from typing import List, Tuple, Dict, Any, Union, Optional

from model.siamese_net import SiameseNet
from data.dataloader import MelodySimDataset

# IMPORTANT: same songs are labelled with 0 in training (means low distance)
class LightningSiameseNet(L.LightningModule):
    def __init__(self, config: Dict):
        super().__init__()
        train_classifier_gap = config["train_classifier_gap"]
        embedding_dim = config["siamese_emb_dim"]

        # Feature extractor trained by triplet loss
        self.siamese_net = SiameseNet(embedding_dim=embedding_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

        # loss functions
        self.criterion_triplet = nn.TripletMarginLoss(margin=2, p=2)
        self.criterion_classification = nn.BCEWithLogitsLoss()

        # how often to train the classification head
        self.train_classifier_gap = train_classifier_gap

        # audio model for inference
        if not hasattr(self, "audio_processor"):
            self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M")
        if not hasattr(self, "audio_model"):
            self.audio_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(self.device)

    def forward_siamese_net(self, anchors, positives, negatives):
        triplet_embeddings = torch.stack([
                self.siamese_net(anchors),
                self.siamese_net(positives),
                self.siamese_net(negatives)
            ], dim=1
        )
        return triplet_embeddings
    
    def forward_classifier(self, triplet_embeddings):
        diff_same = torch.abs(triplet_embeddings[:,0] - triplet_embeddings[:,1])
        diff_diff = torch.abs(triplet_embeddings[:,0] - triplet_embeddings[:,2])
        logit_same = self.classifier(diff_same).squeeze()
        logit_diff = self.classifier(diff_diff).squeeze()
        return logit_same, logit_diff

    def training_step(self, batch, batch_idx):
        anchors, positives, negatives = batch
        B = anchors.shape[0]

        # train siamese net
        triplet_embeddings = self.forward_siamese_net(anchors, positives, negatives)
        loss_triplet = self.criterion_triplet(
            triplet_embeddings[:,0], triplet_embeddings[:,1], triplet_embeddings[:,2]
        ) # anchor_embeddings, positive_embeddings, negative_embeddings

        # train classifier
        train_classifier = False
        if self.train_classifier_gap is None:
            train_classifier = True
        elif self.global_step // self.train_classifier_gap == self.train_classifier_gap - 1:
            train_classifier = True

        if train_classifier:
            labels_same = torch.zeros(B).to(triplet_embeddings.device).float()
            labels_diff = torch.ones(B).to(triplet_embeddings.device).float()
            logit_same, logit_diff = self.forward_classifier(triplet_embeddings.detach())
            loss_same = self.criterion_classification(logit_same, labels_same.squeeze())
            loss_diff = self.criterion_classification(logit_diff, labels_diff.squeeze())
            loss_classification = (loss_same + loss_diff) / 2
        else:
            loss_classification = 0

        # final loss
        final_loss = loss_triplet + loss_classification
        self.log("triplet_loss", loss_triplet)
        self.log("classification_loss", loss_classification)
        self.log("total_loss", final_loss)
        return final_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # B, three = batch.shape[0], batch.shape[1]
        # assert three == 3
        anchors, positives, negatives = batch
        B = anchors.shape[0]

        triplet_embeddings = self.forward_siamese_net(anchors, positives, negatives)
        loss_triplet = self.criterion_triplet(
            triplet_embeddings[:,0], triplet_embeddings[:,1], triplet_embeddings[:,2]
        ).cpu().item() # anchor_embeddings, positive_embeddings, negative_embeddings

        labels_same = torch.zeros(B).to(triplet_embeddings.device)
        labels_diff = torch.ones(B).to(triplet_embeddings.device)
        logit_same, logit_diff = self.forward_classifier(triplet_embeddings)
        loss_same = self.criterion_classification(logit_same, labels_same)
        loss_diff = self.criterion_classification(logit_diff, labels_diff)
        loss_classification = ((loss_same + loss_diff) / 2).cpu().item()

        # ">": normal decision (different song, large logit -> TRUE; same song, small logit -> FALSE)
        # If using "<", then it is reverting the decision

        preds_same = torch.sigmoid(logit_same) > 0.5
        preds_diff = torch.sigmoid(logit_diff) > 0.5
        # preds_same = self._inference_step(batch[:,0], batch[:,1]) > 0.5 # same operation
        # preds_diff = self._inference_step(batch[:,0], batch[:,2]) > 0.5
        preds = torch.cat([preds_same, preds_diff])
        labels = torch.cat([labels_same, labels_diff])
        
        accuracy = (preds.cpu() == labels.cpu()).float().mean()  # Batch accuracy = overall accuracy when batch_size = dataset_size
        accuracy_positive = (preds[:B].cpu() == labels_same.cpu()).float().mean()

        self.log("val_triplet_loss", loss_triplet, prog_bar=True)
        self.log("val_classification_loss", loss_classification, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        self.log("val_accuracy_positive", accuracy, prog_bar=True)

        return {
            "val_triplet_loss": loss_triplet, 
            "val_classification_loss": loss_classification,
        }

    @torch.no_grad()
    def _inference_step(self, sample1:torch.Tensor, sample2:torch.Tensor):
        B1 = sample1.shape[0]
        B2 = sample2.shape[0]
        assert B1 == B2
        B = B1

        out_embs1 = self.siamese_net(sample1)
        out_embs2 = self.siamese_net(sample2)
        diff = torch.abs(out_embs1 - out_embs2)
        logit = self.classifier(diff).squeeze()
        scores = torch.sigmoid(logit)
        return scores

    @torch.no_grad()
    def inference_pairs(
        self,
        waveforms1:Union[List[torch.Tensor], torch.Tensor], 
        waveforms2:Union[List[torch.Tensor], torch.Tensor],
    ):
        time_reduce = torch.nn.AvgPool1d(kernel_size=10, stride=10, count_include_pad=False).to(self.device)
        self.eval()

        if type(waveforms1) == list and type(waveforms2) == list:
            assert len(waveforms1) == len(waveforms2)
            waveforms1 = torch.stack(waveforms1).to(self.device)
            waveforms2 = torch.stack(waveforms2).to(self.device)
        elif torch.is_tensor(waveforms1) and torch.is_tensor(waveforms2):
            assert waveforms1.shape[0] == waveforms2.shape[0]
            assert waveforms1.dim() == 2 and waveforms2.dim() == 2
        else:
            assert 0

        # extract MERT features
        hidden_states1 = self.audio_model(waveforms1, output_hidden_states=True).hidden_states
        hidden_states2 = self.audio_model(waveforms2, output_hidden_states=True).hidden_states
        mert_features1 = torch.stack(
            [time_reduce(h.detach()[:, :, :].permute(0,2,1)).permute(0,2,1) for h in hidden_states1[2::3]], dim=1
        )
        mert_features2 = torch.stack(
            [time_reduce(h.detach()[:, :, :].permute(0,2,1)).permute(0,2,1) for h in hidden_states2[2::3]], dim=1
        )
        batch_num, num_layers, num_frames, layer_dim = mert_features1.shape
        mert_features1 = mert_features1.permute(0, 1, 3, 2) # [batch_num, num_layers=4, layer_dim=768, num_frames]
        mert_features2 = mert_features2.permute(0, 1, 3, 2) # [batch_num, num_layers=4, layer_dim=768, num_frames]
        assert mert_features1.shape[1] == 4 and mert_features1.shape[2] == 768
        # mert_features = mert_features.reshape(batch_num, num_layers * layer_dim, num_frames)
        mert_features1 = torch.cat([mert_features1[:,i] for i in range(mert_features1.shape[1])], dim=1)
        mert_features2 = torch.cat([mert_features2[:,i] for i in range(mert_features2.shape[1])], dim=1)

        # get scores for decisions
        # num_features = mert_features.shape[0] // 2
        scores = self._inference_step(mert_features1, mert_features2)

        return 1 - scores # similarity, the higher the more similar (distance smaller)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            list(self.siamese_net.parameters()) + list(self.classifier.parameters()), 
            lr=1e-3
        )
        return optimizer
