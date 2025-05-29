<div align="center">

# MelodySim - Measuring Melody-aware Music Similarity for Plagiarism Detection

[Model](https://huggingface.co/amaai-lab/MelodySim/tree/main) | [Paper](https://arxiv.org/pdf/2505.20979) | [Dataset](https://huggingface.co/datasets/amaai-lab/melodySim)

We propose MelodySim, a melody-aware music similarity model and dataset for plagiarism detection. First, we introduce a novel method to construct a dataset with focus on melodic similarity. By augmenting Slakh2100; an existing MIDI dataset, we generate variations of each piece while preserving the melody through modifications such as note splitting, arpeggiation, minor track dropout (excluding bass), and re-instrumentation. A user study confirms that positive pairs indeed contain similar melodies, with other musical tracks significantly changed. Second, we develop a segment-wise melodic-similarity detection model that uses a MERT encoder and applies a triplet neural network to capture melodic similarity. The resultant decision matrix highlights where plagiarism might occur. Our model achieves high accuracy on the MelodySim test set.

</div>

## Installation

```bash
git clone https://github.com/AMAAI-Lab/MelodySim.git
cd MelodySim
pip install -r requirements.txt
```

## Dataset

The [MelodySim](https://huggingface.co/datasets/amaai-lab/melodySim) dataset contains 1,710 valid synthesized pieces originated from Slakh2100 dataset, each containing 4 different versions (through various augmentation settings), with a total duration of 419 hours.

## Training
First, pre-compute the MERT features for the melodysim dataset:
```
python data/extract_mert.py -tracks-dir /path/to/melodysim/wav/dataset/train -out-dir /path/to/melodysim/mert/dataset/train
python data/extract_mert.py -tracks-dir /path/to/melodysim/wav/dataset/validation -out-dir /path/to/melodysim/mert/dataset/validation
```
Then, run the training script. The training results (checkpoints) will be saved in the results folder with the following command:
```
python train.py -dataset-root-dir /path/to/melodysim/mert/dataset -result-dir-basename results/training-run
```

## Inference
You can run ```inference.py``` to run the model on two audio files, analyzing their similarity and reaching a decesion on whether or not they are the same song. We provide a positive pair and a negative pair as examples. Try out
```
python inference.py -audio-path1 ./data/example_wavs/Track01968_original.mp3 -audio-path2 ./data/example_wavs/Track01976_original.mp3 -ckpt-path path/to/checkpoint.ckpt
python inference.py -audio-path1 ./data/example_wavs/Track01976_original.mp3 -audio-path2 ./data/example_wavs/Track01976_version1.mp3 -ckpt-path path/to/checkpoint.ckpt
```
Feel free to play around the hyperparameters 
- ```-window-len-sec```, ```-hop-len-sec``` (the way segmenting the input audios);
- ```--proportion-thres``` (how many similar segments should we consider the two pieces to be the same);
- ```--decision-thres``` (between 0 and 1, the smallest similarity value that we consider to be the same);
- ```--min-hits``` (for each window in piece1, the minimum number of similar windows in piece2 to assign that window to be plagiarized).

## Testing
If you are interested in testing performance (precision, recall, F1), you can run the following commands.
```
python test.py -ckpt-path path/to/checkpoint.ckpt -tracks-dir /path/to/melodysim/wav/dataset/test --save-results-dir-basename results/testing --max-num-anchors 78
```
The argument ```--max-num-anchors``` specifies the number of testing anchors. In melodysim dataset, there are 4 versions for each anchor, which means that there are 6 positive pairs for each anchor. ```test.py``` considers all positive pairs plus the "anchor vs. anchor" case; for the negative pairs, it will randomly select the same amount as the positive pairs for testing.

Our testing results are as follows:

|           |**Precision**| **Recall** |   **F1**   |
|-----------|-------------|------------|------------|
| Different | 1.00        | 0.94       | 0.97       |
| Similar   | 0.94        | 1.00       | 0.97       |
| Average   | 0.97        | 0.97       | 0.97       |
| Accuracy  |             |            | 0.97       |

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{lu2025melodysim,
  title={Text2midi-InferAlign: Improving Symbolic Music Generation with Inference-Time Alignment},
  author={Tongyu Lu and Charlotta-Marlena Geist and Jan Melechovsky and Abhinaba Roy and Dorien Herremans},
  year={2025},
  journal={arXiv:2505.20979}
}
```
