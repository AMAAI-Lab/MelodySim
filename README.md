<div align="center">

# MelodySim: Measuring Melody-aware Music Similarity for Plagiarism Detection

[Model](https://huggingface.co/amaai-lab/MelodySim/tree/main) | [Paper](https://arxiv.org/abs/2505.20979) | [Dataset](https://huggingface.co/datasets/amaai-lab/melodySim)

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
To generate new MIDI/audio variants, see [the augmentation guide](data/augmentation/docs/augmentation.md).
The generation environment is separate from the model-training environment. A
complete MIDI-to-WAV demonstration is available with
`python -m data.augmentation.tools.create_augmentation_demo --run` after installing
`requirements-augmentation.txt`.

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


## Augmentation

### Conda (recommended)

Use a separate Python 3.12 environment. Install [Miniforge](https://conda-forge.org/download/)
if conda is unavailable. On Ubuntu/Debian, first install the native libraries:

```bash
sudo apt-get update
sudo apt-get install -y libfluidsynth3 libsndfile1
```

Windows uses the FluidSynth DLLs bundled with `sf2-loader`; macOS native-library
instructions are in the [setup guide](data/augmentation/docs/augmentation-setup.md).
From the repository root, run these commands in a conda-enabled terminal on any platform:

```bash
conda create -n melodysim-augmentation --override-channels -c conda-forge python=3.12 pip -y
conda activate melodysim-augmentation
python -m pip install -r requirements-augmentation.lock
python -m pip check
python -m data.augmentation.tools.check_augmentation
python -m data.augmentation.tools.create_augmentation_demo --run --workers 2
```

For your own data, configure `data/augmentation/configs/slakh.local.yaml` using
the [setup guide](data/augmentation/docs/augmentation-setup.md#7-run-your-dataset),
then run `python -m data.generate_dataset --config data/augmentation/configs/slakh.local.yaml`.

### uv/venv (alternative)

Use Python 3.12 in a separate environment. In PowerShell:

```powershell
uv venv --python 3.12 .venv-augmentation
uv pip install --python .venv-augmentation/Scripts/python.exe -r requirements-augmentation.txt
.venv-augmentation/Scripts/python.exe -m data.augmentation.tools.create_augmentation_demo --run --workers 2
```

The demo writes MIDI variants and WAV clips to `data/augmentation/augmentation-demo/generated`.
For Slakh, set `input_root`, `output_root`, and `soundfont` in a copy of
`data/augmentation/configs/slakh.yaml` (relative paths resolve from the YAML file), then run:

```powershell
.venv-augmentation/Scripts/python.exe -m data.generate_dataset --config data/augmentation/configs/slakh.yaml --dry-run
.venv-augmentation/Scripts/python.exe -m data.generate_dataset --config data/augmentation/configs/slakh.yaml --workers 2
```

On Linux/macOS, use `.venv-augmentation/bin/python` and install the FluidSynth shared
library. See [the augmentation guide](data/augmentation/docs/augmentation.md) for MIDI-only input and other options.


For a fresh machine, follow the [augmentation installation and troubleshooting guide](data/augmentation/docs/augmentation-setup.md)
for conda or uv/venv, native FluidSynth, Musyng SoundFont acquisition/configuration, and self-checks.
Run `python -m data.augmentation.tools.check_augmentation` with your augmentation environment's interpreter
before processing your data.

### Linux with uv/venv (alternative; Ubuntu/Debian)

Run from the repository root. Use a separate Python 3.12 environment for augmentation:

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates libfluidsynth3 libsndfile1

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv --python 3.12 .venv-augmentation
uv pip sync --python .venv-augmentation/bin/python requirements-augmentation.lock
uv pip check --python .venv-augmentation/bin/python

.venv-augmentation/bin/python -m data.augmentation.tools.check_augmentation
.venv-augmentation/bin/python -m data.augmentation.tools.create_augmentation_demo --run --workers 2
```

The demo uses the bundled TimGM6mb SoundFont and needs no external dataset or
Musyng download. Output is under `data/augmentation/augmentation-demo/generated`.

For Slakh, copy `data/augmentation/configs/slakh.yaml` to
`data/augmentation/configs/slakh.local.yaml`, then set `input_root`, `output_root`
and `soundfont`. Relative paths resolve from the YAML file's directory. Run:

```bash
.venv-augmentation/bin/python -m data.generate_dataset --config data/augmentation/configs/slakh.local.yaml --dry-run
.venv-augmentation/bin/python -m data.generate_dataset --config data/augmentation/configs/slakh.local.yaml --workers 2
```

See the [installation guide](data/augmentation/docs/augmentation-setup.md) for
pip installation, Musyng acquisition and troubleshooting. Generation uses the
FluidSynth shared library directly; no sound card or running audio server is needed.
