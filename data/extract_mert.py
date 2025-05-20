import os
import json
import argparse
from typing import Any, Dict, List, Tuple

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from torch.distributed import init_process_group, destroy_process_group  # Import the necessary functions

from data.dataloader import create_audio_dataloader  # Import the create_audio_dataloader function from dataloader.py
import numpy as np

def ddp_setup(rank: int, world_size: int) -> None:
    """Set up the Distributed Data Parallel (DDP) environment.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
    """
    os.environ["MASTER_ADDR"] = "0.0.0.0"
    os.environ["MASTER_PORT"] = "10056"
    init_process_group(backend="nccl", world_size=world_size, rank=rank)

def run_mert_model_and_get_features(waveforms, audio_model, time_reduce=None):
    """Get mert features from waveforms, select only part of layer features

    Args:
        waveforms: torch tensor in shape batch, num_samples)
        audio_model: AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        time_reduce: torch.nn.AvgPool1d(kernel_size=10, stride=10, count_include_pad=False)
    """
    if time_reduce is None:
        time_reduce = torch.nn.AvgPool1d(kernel_size=10, stride=10, count_include_pad=False)
    hidden_states = audio_model(waveforms, output_hidden_states=True).hidden_states
    audio_features = torch.stack(
        [time_reduce(h.detach()[:, :, :].permute(0,2,1)).permute(0,2,1) for h in hidden_states[2::3]], dim=1
    )
    return audio_features
    

def preprocess_mert_features(tracks_dir: str, gpu_id: int, path_template: str) -> None:
    """Preprocess audio features using the MERT model.

    Args:
        gpu_id (int): ID of the GPU to use for processing.
        path_template (str): Template for the output file path.
    """
    print(f"Preprocessing on GPU {gpu_id}")
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M")
    dataloader = create_audio_dataloader(
        tracks_dir=tracks_dir, 
        batch_size=1, 
        num_workers=4, 
        audio_processor=audio_processor
    )

    audio_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(gpu_id)
    time_reduce = torch.nn.AvgPool1d(kernel_size=10, stride=10, count_include_pad=False)

    #source: List[torch.Tensor] = []
    #track_info: List[Dict[str, Any]] = []
    audio_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Processing on GPU {gpu_id}"):
            for track in batch.keys():
                if os.path.exists(path_template.format(track)):
                    continue
                result = []
                for version in batch[track]:
                    waveforms = torch.stack([version[i] for i in range(len(version))]).squeeze().to(gpu_id) # in shape [#segments, #samples]
                    audio_features = run_mert_model_and_get_features(waveforms, audio_model, time_reduce=time_reduce)
                    result.append(audio_features.cpu().numpy()) # in shape [#versions, #segments, #layers=4, #embeddings or seq_len, emb_dim=768]
                
                np.save(path_template.format(track), np.array(result))
                

def main_multiprocess(rank: int, tracks_dir: str, world_size: int, path_template: str) -> None:
    """Main function to run the preprocessing on a single GPU.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        path_template (str): Template for the output file path.
    """
    try:
        ddp_setup(rank, world_size)
        preprocess_mert_features(tracks_dir, rank, path_template)
    except Exception as e:
        print(f"Exception in process {rank}: {e}")
    finally:
        destroy_process_group()

def main(tracks_dir: str, gpu_id: int, path_template: str) -> None:
    """Main function to run the preprocessing on a single GPU.

    Args:
        gpu_id (int): The gpu_id used
        world_size (int): The total number of processes.
        path_template (str): Template for the output file path.
    """
    preprocess_mert_features(tracks_dir, gpu_id, path_template)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MERT features")
    parser.add_argument("-tracks-dir", type=str, help="Data folder storing audio")
    parser.add_argument("-out-dir", type=str, help="Data folder for MERT embedding output")
    parser.add_argument("--gpu-id", type=int, nargs="?", default=None, help="GPU id to use")
    args = parser.parse_args()

    path_template = f"{args.out_dir}/{{0}}.npy"
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    if args.gpu_id is not None:
        main(tracks_dir=args.tracks_dir, gpu_id=args.gpu_id, path_template=path_template)
    else:
        world_size = torch.cuda.device_count()
        print(f"CUDA devices: {world_size}")
        mp.spawn(main_multiprocess, args=(args.tracks_dir, world_size, path_template), nprocs=world_size)