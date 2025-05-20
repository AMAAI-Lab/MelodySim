import os
import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Any, Union, Optional
import time

# Audio dataset
class MelodySimDataset(Dataset):
    def __init__(self, tracks_dir: str, audio_processor=None):
        self.tracks_dir = tracks_dir
        raw_tracklist = os.listdir(tracks_dir)
        self.tracklist = []
        for track in raw_tracklist:
            if "." in track:
                continue
            if len(os.listdir(os.path.join(self.tracks_dir, track))) <= 1:
                continue
            self.tracklist.append(track)
        self.tracklist = sorted(self.tracklist)
        self.audio_processor = audio_processor
        self.sample_rate = None

    def __len__(self) -> int:
        return len(self.tracklist)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        audios = {}
        track = self.tracklist[idx]
        versions = os.listdir(os.path.join(self.tracks_dir, track))
        versions = sorted(versions)

        for version in versions:
            version_path = os.path.join(self.tracks_dir, track, version)
            if "." in version:
                continue
            if not os.path.isdir(version_path):
                continue
            files = os.listdir(version_path)
            valid_files = [f for f in files if f.split('.')[0].isdigit()]
            files = sorted(valid_files, key=lambda x: int(x.split('.')[0]))
            if len(files) == 0:
                continue

            audios[version] = []
            for file in files:
                if ".wav" not in file:
                    # assert 0, f"the audio folder should contain all wav files; folder path: {version_path}"
                    continue
                audio, self.sample_rate = torchaudio.load(os.path.join(self.tracks_dir, track, version, file))
                audios[version].append(audio)
            if self.audio_processor:
                audios[version] = [
                    self.audio_processor(
                        F.resample(waveform, self.sample_rate, self.audio_processor.sampling_rate),
                        sampling_rate=self.audio_processor.sampling_rate,
                        return_tensors="pt")["input_values"].squeeze()
                        for waveform in audios[version]
                ]
        
        if len(audios.keys()) == 0:
            print(f"There is no version stored in track {track}")
        min_frames = min([len(audios[version]) for version in audios.keys()])
        for version in audios.keys():
            audios[version] = audios[version][:min_frames]
        
        return {
            'track': track,
            'audios': audios
        }

def audio_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    '''
    Returns {
        "track_name1": [version1, version2, ...], (each version in shape (num_windows, max_len))
        "track_name2": [version1, version2, ...],
        ...
    }
    '''
    batch_dict = {}
    wav_dict = [item["audios"] for item in batch]
    max_len = 0
    # get max len
    for item in batch:
        for version in item["audios"].keys():
            if item["audios"][version][0].shape[-1] > max_len:
                max_len = item["audios"][version][0].shape[-1]

    # pad the short audio
    for item in batch:
        batch_dict[item["track"]] = []
        for version in item["audios"].keys():
            audio_list = []
            for audio in item["audios"][version]:
                audio_list.append(torch.nn.functional.pad(audio, (0, max_len - audio.shape[-1])))
            batch_dict[item["track"]].append(torch.stack(audio_list))

    return batch_dict

def create_audio_dataloader(
    tracks_dir: str, 
    batch_size: int, 
    num_workers: int, 
    audio_processor=None,
    verbose=True,
) -> DataLoader:
    # print(f'audio_processor is : {audio_processor}')
    dataset = MelodySimDataset(tracks_dir, audio_processor=audio_processor)
    if verbose:
        print("Dataset created. Sample loading:")
        sample_load = dataset[0]
        print(sample_load['track'])
        print(sample_load['audios'].keys())
        print(len(sample_load['audios']["original"]))
        print(sample_load['audios']["original"][0].shape)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False,
        collate_fn=audio_collate_fn, 
        persistent_workers=True
    )
    return dataloader

class MelodysimMERTDataset(Dataset):
    def __init__(self, tracks_dir: str, store_dict: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = os.listdir(tracks_dir)
        self.source = [s for s in self.source if (".ipynb" not in s and "__pycache__" not in s)]
        self.source = sorted(self.source)
        self.tracks_dir = tracks_dir
        self.store_dict = store_dict
        if self.store_dict:
            self.track_dict = self._create_track_dict()
        print(f"Number of samples loaded: {len(self.source)}")  # Add this line to print the number of samples

    def _create_track_dict(self) -> Dict[str, List[int]]:
        track_dict = {}
        print("Loading MERT embeddings, which may take some time...")
        for track in self.source:
            if track not in track_dict:
                track_dict[track] = np.load(os.path.join(self.tracks_dir, track))
        return track_dict

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx: int) -> torch.Tensor:
        trackname = self.source[idx]
        if self.store_dict:
            track = self.track_dict[trackname]
        else:
            track = np.load(os.path.join(self.tracks_dir, trackname))

        assert len(self.source) > 1
        assert track.shape[0] > 1

        num_version, num_frames, num_layers, seq_len, embedding_size = track.shape
        assert num_layers == 4 and embedding_size == 768
        track = torch.from_numpy(track).permute(0, 1, 2, 4, 3)
        track = track.flatten(start_dim=2, end_dim=3)

        return trackname, track # (num_versions, num_frames, embedding_dim, seq_len)

# loading MERT encodings and create triplet instances
class TripletDataset(Dataset):
    def __init__(
        self, 
        tracks_dir: str, 
        store_dict: bool = True, 
        length_mult: int = 4, 
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.source = os.listdir(tracks_dir)
        self.source = [s for s in self.source if (".ipynb" not in s and "__pycache__" not in s)]
        self.tracks_dir = tracks_dir
        self.store_dict = store_dict
        if self.store_dict:
            self.track_dict = self._create_track_dict()
        print(f"Number of samples loaded: {len(self.source)}")  # Add this line to print the number of samples
        self.length_mult = length_mult

    def _create_track_dict(self) -> Dict[str, List[int]]:
        track_dict = {}
        print("Loading MERT embeddings, which may take some time...")
        for track in self.source:
            if track not in track_dict:
                track_dict[track] = np.load(os.path.join(self.tracks_dir, track))
        return track_dict

    def __len__(self):
        return len(self.source) * self.length_mult

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = idx // self.length_mult
        anchor_trackname = self.source[idx]
        if self.store_dict:
            anchor_track = self.track_dict[anchor_trackname]
        else:
            anchor_track = np.load(os.path.join(self.tracks_dir, anchor_trackname))

        assert len(self.source) > 1
        assert anchor_track.shape[0] > 1

        anchor_idx = torch.randint(high=anchor_track.shape[0] - 1, size=()).item()
        time_frame = torch.randint(high=anchor_track.shape[1] - 1, size=()).item()
        anchor_sample = anchor_track[anchor_idx][time_frame]
        
        positive_choices = [i for i in range(anchor_track.shape[0]) if i != anchor_idx]
        positive_idx = random.choice(positive_choices)
        positive_sample = anchor_track[positive_idx][time_frame]

        negative_id_choices = [i for i in range(len(self.source)) if i != idx]
        neg_idx = random.choice(negative_id_choices)
        negative_trackname = self.source[neg_idx]
        if self.store_dict:
            negative_track = self.track_dict[negative_trackname]
        else:
            negative_track = np.load(os.path.join(self.tracks_dir, negative_trackname))
        negative_idx = torch.randint(high=negative_track.shape[0] - 1, size=()).item()
        negative_sample = negative_track[negative_idx][torch.randint(high=negative_track.shape[1] - 1, size=()).item()]

        # concatenate hidden states
        num_layers, seq_len, embedding_size = anchor_sample.shape
        assert num_layers == 4 and embedding_size == 768
        anchor_sample = torch.from_numpy(anchor_sample).permute(0, 2, 1) # (num_layers = 4, embedding_size = 768, seq_len)
        positive_sample = torch.from_numpy(positive_sample).permute(0, 2, 1)
        negative_sample = torch.from_numpy(negative_sample).permute(0, 2, 1)

        anchor = torch.cat([hidden for hidden in anchor_sample], dim=0)
        positive = torch.cat([hidden for hidden in positive_sample], dim=0)
        negative = torch.cat([hidden for hidden in negative_sample], dim=0)

        return anchor, positive, negative # (embedding_dim, seq_len) each
    
def triplet_collate_fn(batch):
    anchors, positives, negatives = tuple(map(list, zip(*batch))) # list[tuples] -> tuple(list)
    max_seq = max([anchor.shape[-1] for anchor in anchors])
    min_seq = min([anchor.shape[-1] for anchor in anchors])
    anchors = torch.stack([torch.nn.functional.pad(anchor, (0,max_seq-anchor.shape[-1]), "constant", 0) for anchor in anchors])

    max_seq = max([positive.shape[-1] for positive in positives])
    min_seq = min([positive.shape[-1] for positive in positives])
    positives = torch.stack([torch.nn.functional.pad(positive, (0,max_seq-positive.shape[-1]), "constant", 0) for positive in positives])

    max_seq = max([negative.shape[-1] for negative in negatives])
    min_seq = min([negative.shape[-1] for negative in negatives])
    negatives = torch.stack([torch.nn.functional.pad(negative, (0,max_seq-negative.shape[-1]), "constant", 0) for negative in negatives])

    return anchors, positives, negatives

def create_triplet_dataloader(
    dataset: TripletDataset, 
    batch_size: int, 
    num_workers: int, 
) -> DataLoader:
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=triplet_collate_fn)
    return loader
    