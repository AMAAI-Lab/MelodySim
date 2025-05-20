import os
import copy
import argparse
import random
import time
from datetime import datetime

import gc
import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from sklearn import metrics
from itertools import combinations
from typing import List, Tuple, Dict, Any, Union, Optional

from config import config
from data.dataloader import MelodySimDataset, MelodysimMERTDataset, create_audio_dataloader
from data.extract_mert import run_mert_model_and_get_features
from model.module import LightningSiameseNet

def window_slide(seq:torch.Tensor, window_len:int, hop_len:int):
    assert len(seq.shape) == 1
    # N, L = seq.shape
    if seq.shape[-1] >= window_len:
        return seq.unfold(-1, window_len, hop_len)
    else:
        return torch.nn.functional.pad(seq, (0, window_len - seq.shape[-1])).unsqueeze(0)

@torch.no_grad()
def inference_two_window_seqs(
    model:LightningSiameseNet,
    audio_model:Optional[AutoModel],
    audio_processor:Optional[Wav2Vec2FeatureExtractor],
    piece1_windowed:Union[List[torch.Tensor], torch.Tensor],
    piece2_windowed:Union[List[torch.Tensor], torch.Tensor],
    batch_size:int = 64
):
    model.eval()

    if type(piece1_windowed) == list and type(piece2_windowed) == list:
        piece1_windowed = torch.stack(piece1_windowed)
        piece2_windowed = torch.stack(piece2_windowed)
    N1, L1 = piece1_windowed.shape
    N2, L2 = piece2_windowed.shape

    audio_model.to(model.device)
    audio_model.eval()
    time_reduce = torch.nn.AvgPool1d(kernel_size=10, stride=10, count_include_pad=False).to(model.device)

    # get MERT encodings
    mert_features1 = run_mert_model_and_get_features(piece1_windowed.to(model.device), audio_model, time_reduce)
    mert_features2 = run_mert_model_and_get_features(piece2_windowed.to(model.device), audio_model, time_reduce)

    # reshape into siamese model input shapes
    num_windows1, num_layers, num_frames1, layer_dim = mert_features1.shape
    num_windows2, num_layers, num_frames2, layer_dim = mert_features2.shape
    assert num_windows1 == N1 and num_windows2 == N2 and num_layers == 4 and layer_dim == 768

    mert_features1 = mert_features1.permute(0, 1, 3, 2) # [num_windows1, num_layers=4, layer_dim=768, num_frames1]
    mert_features2 = mert_features2.permute(0, 1, 3, 2) # [num_windows2, num_layers=4, layer_dim=768, num_frames2]
    assert mert_features1.shape[1] == 4 and mert_features1.shape[2] == 768
    # mert_features1 = mert_features1.reshape(num_windows1, num_layers * layer_dim, num_frames1)
    # mert_features2 = mert_features2.reshape(num_windows2, num_layers * layer_dim, num_frames2)
    mert_features1 = mert_features1.flatten(start_dim=1, end_dim=2)
    mert_features2 = mert_features2.flatten(start_dim=1, end_dim=2)

    L1, L2 = mert_features1.shape[-1], mert_features2.shape[-1]
    N1, N2 = mert_features1.shape[0], mert_features2.shape[0]
    max_len = max(83, max(L1, L2))
    if L1 < max_len:
        mert_features1 = torch.nn.functional.pad(mert_features1, (0, max_len - L1))
    if L2 < max_len:
        mert_features2 = torch.nn.functional.pad(mert_features2, (0, max_len - L2))

    similarity_matrix = torch.zeros(N1, N2)
    for i_row in range(N1):
        similarity_matrix[i_row, :] = 1 - model._inference_step(
            mert_features1[i_row:i_row+1].repeat(N2, 1, 1).to(model.device), 
            mert_features2.to(model.device),
        )
    
    return similarity_matrix # function output 0 means "not plagiarized", 1 means "plagiarized"

@torch.no_grad()
def inference_two_pieces(
    model:LightningSiameseNet,
    audio_model:Optional[AutoModel],
    audio_processor:Optional[Wav2Vec2FeatureExtractor],
    waveform1:Union[np.ndarray, torch.Tensor], 
    waveform2:Union[np.ndarray, torch.Tensor],
    sample_rate:int,
    window_len_sec:float = 10,
    hop_len_sec:float = 10,
    batch_size:int = 64
):
    waveform1_input = audio_processor(
        F.resample(waveform1, sample_rate, audio_processor.sampling_rate),
        sampling_rate=audio_processor.sampling_rate,
        return_tensors="pt"
    )["input_values"].squeeze()

    waveform2_input = audio_processor(
        F.resample(waveform2, sample_rate, audio_processor.sampling_rate),
        sampling_rate=audio_processor.sampling_rate,
        return_tensors="pt"
    )["input_values"].squeeze()

    window_len = int(window_len_sec * audio_processor.sampling_rate)
    hop_len = int(hop_len_sec * audio_processor.sampling_rate)

    waveform1_sec = waveform1_input.shape[-1] / audio_processor.sampling_rate
    waveform2_sec = waveform2_input.shape[-1] / audio_processor.sampling_rate

    piece1_windowed = window_slide(torch.as_tensor(waveform1_input), window_len, hop_len)
    piece2_windowed = window_slide(torch.as_tensor(waveform2_input), window_len, hop_len)

    N1, L1 = piece1_windowed.shape
    N2, L2 = piece2_windowed.shape
    assert L1 == L2 # same window lengths
    L = L1
    
    return inference_two_window_seqs(
        model, audio_model, audio_processor, 
        piece1_windowed, piece2_windowed,
        batch_size=batch_size
    )

def aggregate_decision_matrix(
    similarity_matrix, 
    proportion_thres:float = 0.2, 
    decision_thres:float = 0.5,
    min_hits: int = 1,
    # max_hits_proportion: float = 0.8,
):
    decision_matrix = similarity_matrix > decision_thres
    assert len(decision_matrix.shape) == 2
    N1, N2 = decision_matrix.shape
    row_sum = decision_matrix.sum(dim=0) # for each window in piece 1, how many segments in piece 2 are similar to that seg
    col_sum = decision_matrix.sum(dim=1) # for each window in piece 2, how many segments in piece 1 are similar to that seg
    # row_sum[row_sum > max_hits_proportion * N2] = 0
    # col_sum[col_sum > max_hits_proportion * N1] = 0
    plagiarized_pieces1 = row_sum >= min_hits
    plagiarized_pieces2 = col_sum >= min_hits
    if plagiarized_pieces1.sum() > proportion_thres * N1 and plagiarized_pieces2.sum() > proportion_thres * N2:
        return 1, decision_matrix
    else:
        return 0, decision_matrix

def draft_without_replacement(elements: List, n):
    L = len(elements)
    
    if n <= L:
        return random.sample(elements, n)
    else:
        # Shuffle the elements to randomize order
        shuffled = copy.deepcopy(elements[:])
        random.shuffle(shuffled)
        
        # Repeat the list as needed to get n samples (without exceeding unique elements in each cycle)
        full_draft = shuffled * (n // L) + random.sample(shuffled, n % L)
        return full_draft

@torch.no_grad()
def inference_melodysim_dataset(
    tracks_dir:str,
    model:LightningSiameseNet,
    audio_model:Optional[AutoModel],
    audio_processor:Optional[Wav2Vec2FeatureExtractor],
    max_num_anchors = 6, # due to inference time, only choose a part of anchors for infernece
    proportion_thres:float = 0.2, 
    decision_thres:float = 0.5,
    min_hits:int = 1,
    save_results_dir:Optional[str] = None,
):
    model.eval()
    audio_model.eval()
    dataloader = create_audio_dataloader(tracks_dir=tracks_dir, batch_size=1, num_workers=4, audio_processor=audio_processor)
    data_iter = iter(dataloader)
    dataset_size = len(dataloader)
    num_anchors = max_num_anchors if max_num_anchors < dataset_size else dataset_size
    print(f"Inspecting {num_anchors} pieces")

    # take down all data to inspect
    track_all_versions = []
    track_names = []
    version_keys = []
    for i in range(num_anchors):
        track_data = next(data_iter) # {"track_name1": [version1, ...]}, each version in shape (num_windows, max_len)
        track_names.append(list(track_data.keys())[0])
        track_all_versions.append(track_data[track_names[-1]])
        if len(version_keys) == 0:
            version_keys = [i_ver for i_ver in range(len(track_all_versions[-1]))]

    version_combinations = list(combinations(version_keys, 2))
    negative_combinations = list(combinations(range(num_anchors), 2))
    
    # parse all anchors and construct positive pairs
    positive_pairs = []
    for i in range(num_anchors):
        positive_pairs.append(
            # (i, 'original', i, 'original')
            (i, 0, i, 0)
        )
        for version_pair in version_combinations:
            positive_pairs.append(
                (i, version_pair[0], i, version_pair[1])
            )

    # construct same number of negative pairs as positive pairs for balance
    negative_pairs = []
    negative_pair_pieces = draft_without_replacement(negative_combinations, len(positive_pairs))
    for i, j in negative_pair_pieces:
        version_piece1 = random.choice(version_keys)
        version_piece2 = random.choice(version_keys)
        negative_pairs.append(
            (i, version_piece1, j, version_piece2)
        )
    
    num_positive_pairs = len(positive_pairs)
    num_negative_pairs = len(negative_pairs)
    assert num_positive_pairs == num_negative_pairs
    print(f"Found {num_positive_pairs} positive and negative pairs")

    # do inference for all pairs
    all_pairs = positive_pairs + negative_pairs
    all_labels = [1] * num_positive_pairs + [0] * num_negative_pairs
    decisions = []
    similarity_matrices = []
    decision_matrices = []
    total_inference_time = 0
    for n in range(len(all_pairs)):
        i, ver_i, j, ver_j = all_pairs[n]
        print(f"Comparing {track_names[i]}-{ver_i} and {track_names[j]}-{ver_j}...")

        start_time = time.time()
        piece1_windowed = track_all_versions[i][ver_i]
        piece2_windowed = track_all_versions[j][ver_j]
        N1, N2 = piece1_windowed.shape[0], piece2_windowed.shape[0]
        similarity_matrix = inference_two_window_seqs(model, audio_model, audio_processor, piece1_windowed, piece2_windowed)
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_inference_time += elapsed_time
        print(f"Computation done. Time elapsed {elapsed_time} sec; num of comparisons: {N1}x{N2}={N1*N2}.")

        decision, decision_matrix = aggregate_decision_matrix(
            similarity_matrix, proportion_thres=proportion_thres, decision_thres=decision_thres, min_hits=min_hits
        )
        similarity_matrices.append(similarity_matrix)
        decision_matrices.append(decision_matrix)
        decisions.append(decision)
        print(f"(Decision, Label) = ({decision}, {all_labels[n]})")
        if save_results_dir is not None:
            filename = f"{track_names[i]}-{ver_i} vs {track_names[j]}-{ver_j}.png"
            save_results_path = os.path.join(save_results_dir, filename)
            fig, axes = plt.subplots(1, 2, figsize=(26, 12))
            im1 = axes[0].imshow(decision_matrix.cpu().numpy(), cmap='viridis', aspect='auto', vmin=0, vmax=1)
            axes[0].set_title('Decision Matrix')
            im2 = axes[1].imshow(similarity_matrix.cpu().numpy(), cmap='viridis', aspect='auto', vmin=0, vmax=1)
            axes[1].set_title('Similarity Matrix')
            cbar = fig.colorbar(im2, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)
            cbar.set_label('Color Intensity')
            fig.suptitle(f"(Decision, Label) = ({decision}, {all_labels[n]})")
            plt.savefig(save_results_path, bbox_inches='tight', pad_inches=0, dpi=300)  # Save as high-res image
            plt.close()

        gc.collect()  # Run garbage collection
        torch.cuda.empty_cache()  # Clear unused GPU memory
    
    # get final metric
    print("=== Model Performance Report ===")
    eval_results = metrics.classification_report(
        all_labels, decisions, 
        target_names=["different", "similar"]
    )
    print(eval_results)
    print(f"\n Time elapsed: {total_inference_time} sec.")
    print(f"Num of pieces compared: {len(all_pairs)}")

    save_results_path = os.path.join(save_results_dir, "testing_results.txt")
    with open(save_results_path, "a") as file:
        print(eval_results, file=file)
        print(f"\n Time elapsed: {total_inference_time} sec.", file=file)
        print(f"Num of pieces compared: {len(all_pairs)}", file=file)

@torch.no_grad()
def inference_melodysim_dataset_mert(
    tracks_dir:str,
    model:LightningSiameseNet,
    max_num_anchors = 8, # due to inference time, only choose a part of anchors for infernece
    proportion_thres:float = 0.2, 
    decision_thres:float = 0.5,
    min_hits:int = 1,
    save_results_dir:Optional[str] = None,
):
    model.eval()
    # audio_dataset[i] returns a Tensor of dim (num_versions, num_frames, embedding_dim, seq_len)
    mert_dataset = MelodysimMERTDataset(tracks_dir=tracks_dir)
    dataset_size = len(mert_dataset)
    num_anchors = max_num_anchors if max_num_anchors < dataset_size else dataset_size

    # take down all data to inspect
    track_all_versions = []
    track_names = []
    
    for i in range(num_anchors):
        trackname, trackdata = mert_dataset[i]
        track_names.append(trackname)
        track_all_versions.append(trackdata)

    version_keys = [i for i in range(track_all_versions[0].shape[0])]
    
    version_combinations = list(combinations(version_keys, 2))
    negative_combinations = list(combinations(range(num_anchors), 2))
    
    # parse all anchors and construct positive pairs
    positive_pairs = []
    for i in range(num_anchors):
        positive_pairs.append(
            (i, 0, i, 0)
        )
        for version_pair in version_combinations:
            positive_pairs.append(
                (i, version_pair[0], i, version_pair[1])
            )

    # construct same number of negative pairs as positive pairs for balance
    negative_pairs = []
    negative_pair_pieces = draft_without_replacement(negative_combinations, len(positive_pairs))
    for i, j in negative_pair_pieces:
        version_piece1 = random.choice(version_keys)
        version_piece2 = random.choice(version_keys)
        negative_pairs.append(
            (i, version_piece1, j, version_piece2)
        )
    
    num_positive_pairs = len(positive_pairs)
    num_negative_pairs = len(negative_pairs)
    assert num_positive_pairs == num_negative_pairs
    print(f"Found {num_positive_pairs} positive and negative pairs")

    # do inference for all pairs
    all_pairs = positive_pairs + negative_pairs
    all_labels = [1] * num_positive_pairs + [0] * num_negative_pairs
    decisions = []
    similarity_matrices = []
    decision_matrices = []
    total_inference_time = 0
    for n in range(len(all_pairs)):
        i, ver_i, j, ver_j = all_pairs[n]
        print(f"Comparing {track_names[i]}-{ver_i} and {track_names[j]}-{ver_j}...")

        piece1_merts = track_all_versions[i][ver_i]
        piece2_merts = track_all_versions[j][ver_j]
        L1, L2 = piece1_merts.shape[-1], piece2_merts.shape[-1]
        N1, N2 = piece1_merts.shape[0], piece2_merts.shape[0]
        max_len = max(83, max(L1, L2))
        if L1 < max_len:
            piece1_merts = torch.nn.functional.pad(piece1_merts, (0, max_len - L1))
        if L2 < max_len:
            piece2_merts = torch.nn.functional.pad(piece2_merts, (0, max_len - L2))

        start_time = time.time()
        similarity_matrix = torch.zeros(N1, N2)
        for i_row in range(N1):
            similarity_matrix[i_row, :] = 1 - model._inference_step(
                piece1_merts[i_row:i_row+1].repeat(N2, 1, 1).to(model.device), 
                piece2_merts.to(model.device),
            )
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_inference_time += elapsed_time
        print(f"Computation done. Time elapsed {elapsed_time} sec; num of comparisons: {N1}x{N2}={N1*N2}.")

        decision, decision_matrix = aggregate_decision_matrix(
            similarity_matrix, proportion_thres=proportion_thres, decision_thres=decision_thres, min_hits=min_hits
        )
        similarity_matrices.append(similarity_matrix)
        decision_matrices.append(decision_matrix)
        decisions.append(decision)
        # np.save(f"{i}-{ver_i} vs {j}-{ver_j}.npy", similarity_matrix.cpu().numpy())
        print(f"(Decision, Label) = ({decision}, {all_labels[n]})")
        if save_results_dir is not None:
            filename = f"{track_names[i]}-{ver_i} vs {track_names[j]}-{ver_j}.png"
            save_results_path = os.path.join(save_results_dir, filename)
            fig, axes = plt.subplots(1, 2, figsize=(26, 12))
            im1 = axes[0].imshow(decision_matrix.cpu().numpy(), cmap='viridis', aspect='auto', vmin=0, vmax=1)
            axes[0].set_title('Decision Matrix')
            im2 = axes[1].imshow(similarity_matrix.cpu().numpy(), cmap='viridis', aspect='auto', vmin=0, vmax=1)
            axes[1].set_title('Similarity Matrix')
            cbar = fig.colorbar(im2, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)
            cbar.set_label('Color Intensity')
            fig.suptitle(f"(Decision, Label) = ({decision}, {all_labels[n]})")
            plt.savefig(save_results_path, bbox_inches='tight', pad_inches=0, dpi=300)  # Save as high-res image
            plt.close()

        gc.collect()  # Run garbage collection
        torch.cuda.empty_cache()  # Clear unused GPU memory
    
    # get final metric
    print("=== Model Performance Report ===")
    eval_results = metrics.classification_report(
        all_labels, decisions, 
        target_names=["different", "similar"]
    )
    print(eval_results)
    print(f"\n Time elapsed: {total_inference_time} sec.")
    print(f"Num of pieces compared: {len(all_pairs)}")

    save_results_path = os.path.join(save_results_dir, "testing_results.txt")
    with open(save_results_path, "a") as file:
        print(eval_results, file=file)
        print(f"\n Time elapsed: {total_inference_time} sec.", file=file)
        print(f"Num of pieces compared: {len(all_pairs)}", file=file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piece-wise plagiarism detection")
    parser.add_argument("-ckpt-path", type=str, help="Path to the SiameseNet model checkpoint")
    parser.add_argument("-tracks-dir", type=str, help="Track folder storing wav chunks")
    parser.add_argument("--device", type=str, nargs="?", default=None, help="Specify the device to use")
    parser.add_argument(
        "--save-results-dir-basename", type=str, nargs="?", default=None, 
        help="Folder saving inference results"
    )
    parser.add_argument(
        "--proportion-thres", type=float, default=0.2, 
        help="Proportion threshold of similar windows above which the pair is treated as plagiarized"
    )
    parser.add_argument(
        "--decision-thres", type=float, default=0.5, 
        help="Thres for same/diff decision. The smaller, the more sensitive to same-pair detection."
    )
    parser.add_argument(
        "--min-hits", type=int, default=1, 
        help="For each window in piece1, the minimum number of similar windows in piece2 to assign that window to be plagiarized"
    )
    parser.add_argument(
        "--max-num-anchors", type=int, default=8, 
        help="When running inference on the dataset, how many anchors to choose"
    )
    args = parser.parse_args()

    # get device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # load SiameseNet model
    lightning_module = LightningSiameseNet(config)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    lightning_module.load_state_dict(checkpoint["state_dict"])
    lightning_module.to(device)
    lightning_module.eval()

    # load MERT processor
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M")
    audio_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(device)
    audio_model.eval()

    # get save results dir if given
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    result_dir = args.save_results_dir_basename + "-" + current_time
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    # run inference on dataset
    inference_melodysim_dataset(
        tracks_dir=args.tracks_dir,
        model=lightning_module,
        audio_model=audio_model,
        audio_processor=audio_processor,
        max_num_anchors=args.max_num_anchors,
        proportion_thres=args.proportion_thres, 
        min_hits=args.min_hits,
        save_results_dir=result_dir,
    )

    # optional: load mert dataset instead of loading audio dataset and compute mert on-the-fly
    # inference_melodysim_dataset_mert(
    #     tracks_dir=args.tracks_dir,
    #     model=lightning_module,
    #     max_num_anchors=args.max_num_anchors,
    #     proportion_thres=args.proportion_thres, 
    #     min_hits=args.min_hits,
    #     save_results_dir=result_dir,
    # )
