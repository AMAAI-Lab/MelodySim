import os
import copy
import argparse
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from config import config
from model.module import LightningSiameseNet
from test import inference_two_pieces, aggregate_decision_matrix

SAMPLE_RATE = 44100
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piece-wise plagiarism detection")
    parser.add_argument("-audio-path1", type=str, help="Path to one audio to compare")
    parser.add_argument("-audio-path2", type=str, help="Path to another audio to compare")
    parser.add_argument("-ckpt-path", type=str, help="Path to the SiameseNet model checkpoint")
    parser.add_argument("--device", type=str, nargs="?", default=None, help="Specify the device to use")
    parser.add_argument(
        "--save-results-dir", type=str, nargs="?", default="", 
        help="Folder saving inference results"
    )
    parser.add_argument(
        "-window-len-sec", type=float, default=10, 
        help="Window lengh chunking audio into windows"
    )
    parser.add_argument(
        "-hop-len-sec", type=float, default=10,
        help="Hop lengh chunking audio into windows"
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

    # read audios
    wav1, sr = torchaudio.load(args.audio_path1, normalize=True, channels_first=True)
    wav1 = wav1.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav1 = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(wav1)

    wav2, sr = torchaudio.load(args.audio_path2, normalize=True, channels_first=True)
    wav2 = wav2.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav2 = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(wav2)
    
    # get similarity matrix
    similarity_matrix = inference_two_pieces(
        model=lightning_module,
        audio_model=audio_model,
        audio_processor=audio_processor,
        waveform1=wav1, 
        waveform2=wav2,
        sample_rate=SAMPLE_RATE,
        window_len_sec=args.window_len_sec,
        hop_len_sec=args.hop_len_sec
    )

    # get decision
    decision, decision_matrix = aggregate_decision_matrix(
        similarity_matrix, 
        proportion_thres=args.proportion_thres, 
        decision_thres=args.decision_thres, 
        min_hits=args.min_hits
    )
    decision_strings = {0: "not same piece", 1: "same piece"}
    decision_string = decision_strings[decision]

    # output inference results
    audioname1 = os.path.splitext(os.path.basename(args.audio_path1))[0]
    audioname2 = os.path.splitext(os.path.basename(args.audio_path2))[0]

    filename = f"inference_{audioname1}_vs_{audioname2}.png"
    save_path = os.path.join(args.save_results_dir, filename)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    im1 = axes[0].imshow(decision_matrix.cpu().numpy(), cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title('Decision Matrix')
    im2 = axes[1].imshow(similarity_matrix.cpu().numpy(), cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title('Similarity Matrix')
    cbar = fig.colorbar(im2, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)
    cbar.set_label('Color Intensity')
    fig.suptitle(f"Decision = {decision} ({decision_string})")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    print(f"Inference finished, decision: {decision_string}")
    print(f"Similarity matrix saved to {save_path}")