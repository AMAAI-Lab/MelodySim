# Augmentation installation and troubleshooting

This guide starts from a fresh machine and ends with generated MIDI variants and
WAV clips. Run every project command **from the MelodySim repository root**.
Use a dedicated **64-bit CPython 3.12** environment. A GPU, MERT weights, an audio
device, and the model-training requirements are not needed for generation.

- [System libraries](#1-system-libraries)
- [Install with conda](#2-install-with-conda-recommended)
- [Install with uv/venv](#3-install-with-uvvenv-alternative)
- [Install with standard venv/pip](#4-install-with-standard-venvpip-alternative)
- [Self-check and complete demo](#5-self-check-and-complete-demo)
- [Acquire and configure Musyng](#6-acquire-and-configure-musyng)
- [Run your dataset](#7-run-your-dataset)
- [Troubleshooting](#troubleshooting)
- [Validation and release contents](#validation-and-release-contents)

## 1. System libraries

Install [Git](https://git-scm.com/downloads), then get the source if needed:

```sh
git clone https://github.com/AMAAI-Lab/MelodySim.git
cd MelodySim
```

There are three separate dependencies: Python packages, the **FluidSynth native
library** that synthesizes audio, and a **SoundFont `.sf2` data file** containing
instrument samples. Installing a Python binding does not install Musyng.
The recommended conda route manages Python and pip; install the native libraries
below on Linux/macOS. Windows uses the bundled DLLs. A conda environment alone
does not supply those system libraries or a SoundFont.

### Windows 10/11, x64

`sf2-loader==1.29` supplies its FluidSynth DLLs. Installing the Python dependencies
below is normally sufficient; no system FluidSynth installation, virtual MIDI
driver or DLL copying is needed. Python and DLL architectures must match. Run the
self-check to verify the actual library. See the upstream
[sf2_loader installation notes](https://github.com/Rainbow-Dreamer/sf2_loader#installation).

### Ubuntu 22.04/24.04 or Debian 12, x64

```sh
sudo apt-get update
sudo apt-get install -y git curl ca-certificates libfluidsynth3 libsndfile1
```

`libfluidsynth3` is the shared library, even though the library implements
FluidSynth 2.x. The similarly named Python package is not a replacement. The
`fluidsynth` command-line executable is optional for this pipeline. Package details:
[Ubuntu libfluidsynth3](https://packages.ubuntu.com/noble/libfluidsynth3),
[FluidSynth distributions](https://www.fluidsynth.org/download/).

### macOS

Install [Homebrew](https://brew.sh/) and run:

```sh
brew install git fluid-synth libsndfile
```

Keep Python and Homebrew libraries on the same architecture; avoid mixing an
Intel/Rosetta Python with ARM libraries. If discovery fails, set the library search
path in the terminal used for generation and retry the self-check:

```sh
export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix fluid-synth)/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"
```

Reference: [Homebrew fluid-synth formula](https://formulae.brew.sh/formula/fluid-synth).

This project reads WAV/FLAC through SoundFile and writes PCM WAV directly. **ffmpeg
is optional for this generation path**, even though the upstream musicpy/pydub
stack can warn about its absence. MP3 conversion and model inference have separate
requirements. No audio server or sound card needs to be configured.

## 2. Install with conda (recommended)

Install [Miniforge](https://conda-forge.org/download/) for your operating system
if conda is not already available. An existing Miniconda/Anaconda installation
also works; create a dedicated environment rather than modifying `base`.
On Windows, download the x86-64 installer, install for your user and open the
Miniforge Prompt (or the conda-enabled prompt from your existing installation).
On Linux/macOS, a fresh user installation can be bootstrapped from the repository
root with the official installer for your OS/architecture:

```sh
mkdir -p .cache/installers
curl -fL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -o .cache/installers/miniforge.sh
bash .cache/installers/miniforge.sh -b -p "$HOME/miniforge3"
. "$HOME/miniforge3/etc/profile.d/conda.sh"
```

Skip the installer if you already have conda. To enable it automatically in later
terminals, run `conda init bash` or `conda init zsh` for your shell and reopen the
terminal. Then, from the repository root:

```sh
conda --version
conda create -n melodysim-augmentation --override-channels -c conda-forge python=3.12 pip -y
conda activate melodysim-augmentation
python -m pip install -r requirements-augmentation.lock
python -m pip check
python -m data.augmentation.tools.check_augmentation
python -m data.augmentation.tools.create_augmentation_demo --run --workers 2
```

The activated environment uses `python` on Windows, Linux and macOS; there is no
`.venv-augmentation/Scripts` versus `.venv-augmentation/bin` distinction. No venv
inside the conda environment is needed. `--override-channels -c conda-forge`
selects the channel for this environment without changing global channel settings.
Conda provides Python/pip, while pip installs the tested Python package snapshot,
including scikit-learn 1.5.2 and setuptools below 81. Follow section 1 for native
FluidSynth/SoundFile libraries and section 6 for Musyng.

Install any required conda packages **before** the pip requirements. When changing
the conda package set later, prefer a fresh environment, then reinstall the pip
snapshot and run the self-check; see the official
[conda/pip guidance](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#using-pip-in-an-environment).
Use `python -m pip` inside the activated environment, without `--user` or `sudo`.
Keep the original model-training `requirements.txt` in a different environment.

For a scheduler or shell without activation support, use `conda run` instead:

```sh
conda run --no-capture-output -n melodysim-augmentation python -m data.augmentation.tools.check_augmentation
conda run --no-capture-output -n melodysim-augmentation python -m data.generate_dataset --config data/augmentation/configs/slakh.local.yaml --workers 2
```

The second command assumes your local dataset YAML is configured as in section 7.
If an existing demo was generated in a different Python environment, choose a new
demo root, for example `python -m data.augmentation.tools.create_augmentation_demo
--root outputs/augmentation-demo-conda --run --workers 2`. Provenance changes
correctly prevent silently reusing outputs from a different environment.

## 3. Install with uv/venv (alternative)

Install [uv using its official installer](https://docs.astral.sh/uv/getting-started/installation/).
It can also [install Python 3.12](https://docs.astral.sh/uv/guides/install-python/),
so an existing Python installation is not necessary.

### Windows PowerShell

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new terminal, return to the repository root, then run:

```powershell
uv --version
uv python install 3.12
uv venv --python 3.12 .venv-augmentation
uv pip sync --python .venv-augmentation/Scripts/python.exe requirements-augmentation.lock
uv pip check --python .venv-augmentation/Scripts/python.exe
.venv-augmentation/Scripts/python.exe -m data.augmentation.tools.check_augmentation
.venv-augmentation/Scripts/python.exe -m data.augmentation.tools.create_augmentation_demo --run --workers 2
```

### Linux/macOS shell

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal, return to the repository root, then run:

```sh
uv --version
uv python install 3.12
uv venv --python 3.12 .venv-augmentation
uv pip sync --python .venv-augmentation/bin/python requirements-augmentation.lock
uv pip check --python .venv-augmentation/bin/python
.venv-augmentation/bin/python -m data.augmentation.tools.check_augmentation
.venv-augmentation/bin/python -m data.augmentation.tools.create_augmentation_demo --run --workers 2
```

Environment activation is optional: these commands select the intended interpreter
explicitly and work without changing the PowerShell activation-script policy.
`uv pip` does not require pip to be installed inside the environment.

`requirements-augmentation.lock` pins the package snapshot used for release
validation, including pytest. It is a requirements-format snapshot, not a uv
project lock; use `uv pip sync`, not `uv sync`. It does not pin native libraries or
SoundFonts. `requirements-augmentation.txt` lists the direct runtime requirements
and allowed ranges. To resolve a newer dependency set instead of the snapshot:

```sh
uv pip install --python .venv-augmentation/bin/python -r requirements-augmentation.txt
```

On Windows substitute `.venv-augmentation/Scripts/python.exe`. Rerun the self-check
and tests after changing package versions. Keep scikit-learn **1.5.2** for the
bundled classifier and setuptools **below 81** for pretty_midi's `pkg_resources`
import. The original `requirements.txt` serves model training/inference and pins a
different scikit-learn version; install it in a different environment.

## 4. Install with standard venv/pip (alternative)

Start with a 64-bit CPython 3.12 installation that includes `venv` and `pip`, from
[Python.org](https://www.python.org/downloads/) or your operating-system package
manager. On Ubuntu 24.04, `sudo apt-get install python3.12 python3.12-venv` supplies
the needed interpreter and environment module. If your distribution does not
provide Python 3.12, use the uv route above.

### Windows PowerShell

```powershell
py -3.12 --version
py -3.12 -m venv .venv-augmentation
.venv-augmentation/Scripts/python.exe -m pip install --upgrade pip
.venv-augmentation/Scripts/python.exe -m pip install -r requirements-augmentation.lock
.venv-augmentation/Scripts/python.exe -m pip check
.venv-augmentation/Scripts/python.exe -m data.augmentation.tools.check_augmentation
.venv-augmentation/Scripts/python.exe -m data.augmentation.tools.create_augmentation_demo --run --workers 2
```

If the Python launcher `py` is unavailable, use the absolute path of your Python
3.12 executable for the first two commands.

### Linux/macOS shell

```sh
python3.12 --version
python3.12 -m venv .venv-augmentation
.venv-augmentation/bin/python -m pip install --upgrade pip
.venv-augmentation/bin/python -m pip install -r requirements-augmentation.lock
.venv-augmentation/bin/python -m pip check
.venv-augmentation/bin/python -m data.augmentation.tools.check_augmentation
.venv-augmentation/bin/python -m data.augmentation.tools.create_augmentation_demo --run --workers 2
```

Use a fresh environment: `pip install -r` does not remove conflicting extra packages
left over from training. To install only the direct runtime requirements, replace
the lock filename with `requirements-augmentation.txt`; install pytest separately
if running tests. See the official [venv documentation](https://docs.python.org/3.12/library/venv.html).

## 5. Self-check and complete demo

The default self-check and demo use `TimGM6mb.sf2` bundled with pretty_midi 0.2.10.
No external SoundFont download or Slakh data is required for this first run.
The small SoundFont is for functional validation; it does not reproduce Musyng's
timbre or the paper's dataset.

`data.augmentation.tools.check_augmentation` exits with code 0 only after checking Python/package
versions, loading the classifier, enumerating all 128 GM melodic programs and the
standard drum kit, synthesizing MIDI, applying pitch/time/tempo changes, and
round-tripping non-silent WAV/FLAC audio. It prints `[OK]` lines, the native
FluidSynth version and the SoundFont SHA-256. Results are written to:

```text
outputs/augmentation-check/
  report.json
  self_check.mid
  self_check.wav
  self_check.flac
```

Rerunning replaces these diagnostic files. Use `--output-dir PATH` for a different
location. A failure prints `[FAIL]`, exits with code 1, and should be resolved
before processing a dataset. The self-check does not inspect your input dataset;
use generation's `--dry-run` for that.

The demo builds six toy sources, two each in train/validation/test, then generates
the original plus three augmented versions of each. Expect **24 MIDI files and
48 WAV clips**, a top-level `dataset_manifest.json` with `status: complete`, and a
`transformations.json` under each source:

```text
data/augmentation/augmentation-demo/
  demo.yaml
  source/
  generated/
    dataset_manifest.json
    train/Demo00/original.mid
    train/Demo00/version_0.mid
    train/Demo00/original/0.wav
    train/Demo00/version_0/0.wav
    ...
```

Rerunning verifies and skips matching complete tracks. The first librosa/Numba
invocation can take longer while compilation caches are built. Start with
`--workers 1` when diagnosing a problem.

## 6. Acquire and configure Musyng

### Choose the asset deliberately

The paper calls its SoundFont "Musyng". The old server's exact file and hash are
lost. **Musyng and Musyng Kite are different releases**; downloading Kite does not
establish that it matches the paper. The author's
[release thread](https://www.kvraudio.com/forum/viewtopic.php?t=351893) distinguishes
them and recommends Kite. The usable public route below is **Musyng Kite**. If
you recover the historical Musyng file, point the same configuration to it instead.

The author also [announced the SynthFont-hosted mirror](https://www.kvraudio.com/forum/viewtopic.php?p=5766421).
The following endpoints were checked on 2026-10-15: the SoundFont archive and
SFPACK ZIP responded successfully; the old `1drv.ms/VeywIN` link returned 404.
Download links can change, so keep the upstream thread as the source reference.

### Download and extract

Download [Musyng_Kite.sfpack](https://www.synthfont.com/SoundFonts/Musyng_Kite.sfpack)
(about 354 MB compressed). It must be **decompressed to `.sf2`**. Renaming `.sfpack`
to `.sf2`, or unpacking it with ordinary ZIP/7-Zip tools, does not perform that
conversion. Keep several GB of free disk space for the archive, extracted font
and generated audio.

Windows PowerShell, from the repository root:

```powershell
New-Item -ItemType Directory -Force assets | Out-Null
Invoke-WebRequest https://www.synthfont.com/SoundFonts/Musyng_Kite.sfpack -OutFile assets/Musyng_Kite.sfpack
Invoke-WebRequest https://www.synthfont.com/SFPACK.zip -OutFile assets/SFPACK.zip
Expand-Archive -LiteralPath assets/SFPACK.zip -DestinationPath assets/sfpack-tool
```

Open the extracted `SFPACK.exe`, choose **File > Add Files** (or the Add Files
toolbar button), select `Musyng_Kite.sfpack`, and start the decompression with
**Go!**. Keep the resulting `.sf2` in `assets/` and retain its actual filename;
the examples below use `assets/Musyng_Kite.sf2`. If the tool writes a name with
spaces, use that quoted path instead. The tool is linked from
[SynthFont's SoundFont page](https://www.synthfont.com/soundfonts.html).

On Linux/macOS, downloading uses the same archive:

```sh
mkdir -p assets
curl -fL --retry 3 https://www.synthfont.com/SoundFonts/Musyng_Kite.sfpack -o assets/Musyng_Kite.sfpack
```

SFPACK is a Windows utility. A straightforward route for Linux/macOS or headless
servers is to extract it on a Windows machine using the steps above, then copy
the resulting `.sf2` into `assets/` on the target machine. An existing Wine setup
can also run the utility, but this release does not validate Wine installation or
SFPACK under Wine. If no extractor is available, continue with the bundled TimGM
demo; the pipeline cannot load an unextracted SFPACK archive. A MIDI.js
`MusyngKite/` folder of JavaScript/MP3 samples is also not an SF2 file.

### Verify and record the file

PowerShell:

```powershell
python -m data.augmentation.tools.check_augmentation --soundfont assets/Musyng_Kite.sf2 --output-dir outputs/musyng-check
Get-FileHash -Algorithm SHA256 -LiteralPath assets/Musyng_Kite.sf2
```

Linux/macOS:

```sh
python -m data.augmentation.tools.check_augmentation --soundfont assets/Musyng_Kite.sf2 --output-dir outputs/musyng-check
```

The report records SHA-256, preset coverage and the real synthesis result. Copy
the SHA-256 with your experiment records and compare it when transferring the
file to another machine. It identifies your file; it is **not** a published
checksum of the lost paper asset. Preserve the upstream download URL, version
notes and license/credits. Musyng Kite's author discusses its
[CC BY-SA 3.0 release in the upstream thread](https://www.kvraudio.com/forum/viewtopic.php?start=135&t=351893);
the repository's code license does not replace SoundFont terms. No external
SoundFont or SFPACK executable is distributed in this repository.

## 7. Run your dataset

Copy the template to `data/augmentation/configs/slakh.local.yaml` (ignored by Git), then
edit its paths. Keep the copy in the same directory for these relative paths:

```yaml
input_root: ../../../datasets/slakh2100_flac_redux
output_root: ../../../outputs/augmentation/slakh-kite
soundfont: ../../../assets/Musyng_Kite.sf2
original_mode: audio_mix
```

Relative paths are resolved **from the YAML file**, including command-line path
overrides. Use forward slashes in YAML, or single-quote Windows paths such as
`'E:\datasets\slakh'`. Input and output roots must be separate and non-nested.

Expected source layout:

```text
datasets/slakh2100_flac_redux/
  train/Track00001/all_src.mid
  train/Track00001/mix.flac
  validation/Track01001/all_src.mid
  validation/Track01001/mix.flac
  test/Track02001/all_src.mid
  test/Track02001/mix.flac
```

The project does not ship or recover Slakh. Obtain it through the
[Slakh dataset project](https://www.slakh.com/), preserve its source-piece splits,
and check the extracted directory layout. For multitrack MIDI without source
audio, set `original_mode: synthesize`; this also renders the unmodified original.
For a flat folder of MIDI files, additionally set `split: train` (or another
intended split). In `audio_mix` mode, flat MIDI files need sibling `<stem>.wav`.

With the conda environment active, after saving your local YAML:

```powershell
python -m data.augmentation.tools.check_augmentation --config data/augmentation/configs/slakh.local.yaml
python -m data.generate_dataset --config data/augmentation/configs/slakh.local.yaml --dry-run
python -m data.generate_dataset --config data/augmentation/configs/slakh.local.yaml --limit 1 --workers 1
python -m data.generate_dataset --config data/augmentation/configs/slakh.local.yaml --workers 2
```

These commands also work on Linux/macOS with the conda environment active. `--dry-run`
checks configuration, assets, input discovery and split isolation; the one-source
run checks real generation. Check `dataset_manifest.json` for `status: complete`
after the full run. Begin with one worker for a large SoundFont: every worker may
load its own copy, in addition to whole-piece audio processing buffers.

See [the pipeline guide](augmentation.md) for melody overrides, augmentation
policies, replay, output layout, MERT integration and differences from the paper.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `conda` is not recognized / command not found | Install Miniforge for your architecture, or open the conda-enabled prompt from your existing installation. On Linux/macOS initialize the shell using the installer instructions. |
| `conda activate` says the shell is not initialized | Run `conda init bash`, `conda init zsh` or `conda init powershell` for your shell, then reopen the terminal. For batch jobs use `conda run --no-capture-output -n melodysim-augmentation python ...`. |
| The wrong Python/package versions appear after switching environments | Run `conda activate melodysim-augmentation` and check `python -c "import sys; print(sys.executable)"`. Install with that Python's `-m pip`; do not nest a venv or use `--user`. |
| `uv` is not recognized / command not found | Open a new terminal after installing; check the installation location reported by the official installer and add that directory to PATH. Alternatively use the pip route. |
| `No module named pip` in a uv environment | Use `uv pip ... --python <env-python>` directly. To switch to pip, run `<env-python> -m ensurepip --upgrade` first. |
| `ensurepip` / `venv` unavailable on Ubuntu | Install `python3.12-venv` for that interpreter, or use uv-managed Python. |
| `externally-managed-environment` | Create the dedicated virtual environment and invoke its Python. Do not install this stack into the OS Python. |
| `No module named data` | Change directory to the repository root; use `python -m data.generate_dataset` and `python -m data.augmentation.tools.check_augmentation`, not a direct script path. |
| `audioop` / `pyaudioop` missing, or unsupported Python message | Use CPython 3.12. The pinned pydub stack relies on audioop, which was removed from Python 3.13. |
| `pkg_resources` missing | Reinstall the locked dependencies; pretty_midi 0.2.10 needs `setuptools<81`. Its deprecation warning alone does not prevent a successful check. |
| scikit-learn version warning, pickle load failure, wrong classifier features | Use the dedicated environment with scikit-learn 1.5.2 and the repository classifier. Do not mix the training requirements into it. Verify `classifier_model.pkl` exists with the SHA-256 recorded in the adjacent JSON. |
| `Cannot load sf2_loader/FluidSynth`, `libfluidsynth*.so`/`.dylib` missing | Install the native library from section 1. On macOS apply the documented library search path. Recheck architecture and run the self-check in the same terminal. Installing `pyfluidsynth` alone does not install the library. |
| Windows `DLL load failed`, `WinError 126/193` | Check 64-bit Python and reinstall `sf2-loader==1.29` in a fresh environment. Use its bundled DLLs; remove manually added conflicting DLL paths. The error can also mean a DLL dependency is missing; inspect the underlying error before replacing libraries. |
| ffmpeg warning, pydub invalid-escape/deprecation warning | These upstream warnings can appear while generation still succeeds. WAV/FLAC self-check success verifies the required path; ffmpeg is only needed for additional conversion workflows. |
| `Expected an extracted RIFF SoundFont`, `Cannot load soundfont`, or `SoundFont is truncated` | Extract `.sfpack` with SFPACK. Ensure the file is a real `.sf2`, not an HTML error page, renamed archive or incomplete download; rerun `--soundfont` self-check. |
| Missing GM preset / drum kit | Use a complete GM SoundFont with programs 0–127 in bank 0 and drum kit 0 in bank 128. A piano-only font cannot render this dataset. |
| File not found despite a valid-looking path | Resolve the path from the YAML's directory, not the shell directory. Quote spaces in shell arguments. For YAML use forward slashes or single quotes around Windows backslashes. |
| Missing `mix.flac` / source audio | Check the Slakh layout; choose `original_mode: synthesize` deliberately for MIDI-only sources. |
| No pitched track / no sources found | Check `all_src.mid`, input root, split directories, or explicit `split` for a flat input. Percussion-only MIDI cannot supply melody. |
| Duplicate ID or MIDI across splits | Fix the source split allocation; all versions of one piece must stay in one split. |
| Changed fingerprint/settings; resume refuses output | Keep a new output directory for changed assets, dependencies or settings. Use `--overwrite` only when replacing recognized generated tracks is intended. |
| Another generator is using output root | Wait for the running process, or choose a different output root. The OS releases the lock when a process exits; a leftover lock file alone is not an active lock. |
| Workers terminate / out of memory | Retry with `--workers 1`, the small bundled font and `--limit 1`; then increase resources or workers gradually. Large fonts and whole-piece audio buffers consume memory per worker. |
| Network timeout or package unavailable | Check access to PyPI, GitHub/Python downloads and SynthFont. If using an institutional mirror, confirm it carries the exact pinned versions. Do not silently substitute a different SoundFont or classifier version. |

If a run fails, keep its terminal error, `report.json` if produced, the local YAML
and the dataset/track manifests. A generation failure returns a nonzero exit code;
read the failed source entries in `dataset_manifest.json`. A previous successful
diagnostic report is not evidence that a later failed check succeeded: use the
current command's exit code.

## Validation and release contents

Run the regression suite using your augmentation interpreter:

```powershell
python -m pytest data/augmentation/tests -q --basetemp=.cache/pytest
```

With conda, activate `melodysim-augmentation` first. For the alternative venv setup,
use its interpreter path. The lock includes pytest;
the direct runtime requirements do not. Model-loader boundary tests skip unless
torch and torchaudio are also available. They are optional for augmentation; the
[pipeline guide](augmentation.md#validation) explains their separate CPU install.
With torch installed, run `python -m pytest -q --basetemp=.cache/pytest` to include
the model aggregation regressions in `model/tests/` as well. Those tests exercise
both orientations of unequal window counts, directional coverage, `min_hits`,
and the existing strict threshold boundaries; they do not validate the historical
MCIC pair construction or reproduce the published score.

Windows x64 / CPython 3.12 is the executed release-validation platform. Linux/macOS
instructions follow upstream installation documentation; this repository does not
claim they were executed on those operating systems. Native versions and hardware
can change rendered bytes; package pins alone do not guarantee cross-platform PCM
identity. Generation records package, native backend, source and asset identities.

Only the supported generator, classifier, configuration template, demo/self-check,
tests and documentation belong in the augmentation release. Legacy generators,
notebooks, old environment exports, SoundFonts, local datasets, virtual
environments, caches and rendered outputs are excluded. The classifier's metadata
keeps its historical origin even though the old source tree is no longer shipped.
