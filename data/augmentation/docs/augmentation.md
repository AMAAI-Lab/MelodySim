# Offline augmentation

This pipeline builds melody-aware variants from multitrack MIDI. It writes the
directory layout consumed by `data.dataloader.MelodySimDataset` and
`data.extract_mert`, without loading MERT or requiring a GPU. It is a new,
versioned implementation, not a byte-for-byte reconstruction of the historical
MelodySim dataset. The old server's source assets and random states are lost.

## Install and run a complete demonstration

For a new machine, follow [the installation and troubleshooting guide](augmentation-setup.md).
It covers uv and pip, native FluidSynth, Musyng acquisition/extraction, SoundFont
configuration, and an executable self-check. Commands below assume that environment
is installed; use its Python executable rather than the model-training environment.

Use a separate Python 3.12 environment. The bundled melody classifier was trained
with scikit-learn 1.5.2; the model-training requirements currently pin a different
version. Do not install the two requirements files together in the same environment.

```powershell
uv venv --python 3.12 .venv-augmentation
uv pip sync --python .venv-augmentation/Scripts/python.exe requirements-augmentation.lock
.venv-augmentation/Scripts/python.exe -m data.augmentation.tools.create_augmentation_demo --run --workers 2
```

On Linux/macOS use `.venv-augmentation/bin/python` and install the system FluidSynth shared
library (for example `libfluidsynth3` on Debian/Ubuntu). Windows DLLs are supplied
by `sf2-loader`. No sound card is required. The locked package set used for
validation is in `requirements-augmentation.lock` and can be installed with
`uv pip sync --python <venv-python> requirements-augmentation.lock`.

The demo creates six original toy compositions, two per split, and renders their
originals plus three variants using **TimGM6mb.sf2 bundled with pretty_midi**.
It exercises the real classifier, SoundFont engine, all augmentation stages and
WAV writer. The test SoundFont is not Musyng and these compositions are not Slakh.
The SoundFont stays in the installed package; no copy is added to this repository.

Output is under `data/augmentation/augmentation-demo/generated`. Run the same command again
to check and resume existing output. A demo configuration is written to
`data/augmentation/augmentation-demo/demo.yaml`.

## Use a real dataset

Copy `data/augmentation/configs/slakh.yaml` and set `input_root`, `output_root`, and
`soundfont`. Relative paths, including CLI path overrides, are resolved against
the configuration file's directory. Absolute paths also work. Input and output
directories must be separate, non-nested locations.

```text
input_root/
  train/Track00001/all_src.mid
  train/Track00001/mix.flac
  validation/Track01001/all_src.mid
  validation/Track01001/mix.flac
  test/Track02001/all_src.mid
  test/Track02001/mix.flac
```

```powershell
python -m data.generate_dataset --config path/to/slakh.yaml --dry-run
python -m data.generate_dataset --config path/to/slakh.yaml --workers 2
```

`--limit N` selects the first N sources in deterministic split/name order, for
smoke testing. For a flat directory of `.mid` files or Slakh track directories,
set `split: train` (or pass `--split train`). A flat MIDI's original audio is read
from its sibling `<stem>.wav` in `audio_mix` mode. With `original_mode: synthesize`
no source audio is needed; the unmodified MIDI is rendered with the same SoundFont.
Missing source files fail explicitly rather than silently changing modes.

Track IDs must use ASCII letters, digits, hyphens or underscores. Their names
remain unchanged. Duplicate IDs or identical MIDI hashes across splits are
rejected. Track order and numeric clip names are deterministic.

## What happens to each source

1. Read and sort the original MIDI. Identify melody with the bundled 18-feature
   gradient boosting pipeline **before** altering its notes or instruments.
2. Sample independent per-track dropout and note-operation probabilities. Protect
   the identified melody, bass, piano, guitar and vocal tracks from dropout.
3. Select replacement instruments, split eligible notes, invert eligible block
   chords, and arpeggiate them, following Fig. 1's note-stage order. All edits use PrettyMIDI's zero-based GM programs
   and second-based timing.
4. Render the original/augmented MIDI events through the FluidSynth binding
   supplied by `sf2-loader`. Note-off, controller, bend and note-on ordering is
   explicit. Every track receives its own channel, including multiple drum tracks.
   Sustain, dynamics and pitch bends are retained; floating-point synthesis avoids
   clipping dense mixes before peak scaling. Bank changes are fixed to GM
   bank 0 / percussion bank 128. Reverb and chorus are disabled for reproducibility.
5. Use the original waveform's length as the canonical timeline. Pad or trim each
   rendered variant to that length. Pitch-shift the whole waveform, apply a
   whole-piece circular time shift, then slice windows. Apply tempo processing
   to each window, then write mono
   PCM-16 WAVs at the configured rate. Peaks above 0.98 are scaled, and that scale
   is recorded.

`original_mode: audio_mix` preserves the Slakh mix as the original, resampling and
mixing channels if needed; only variants go through the SoundFont renderer.
`synthesize` also renders the original. The chosen mode and source hashes are in
every track manifest.

## Explicit musical choices and historical differences

These reviewed rules are versioned as `melodysim-augmentation-v2`. Earlier v1
outputs must use a new output directory or explicit `--overwrite`; their plans
are not replay-compatible with the changed processing order and note targets.

| Rule | Current behavior and reason |
| --- | --- |
| Melody selection | One highest-scoring nonempty, non-drum track by default. Optional manual lists support multiple melody tracks. Identification is cached once per source. |
| Classifier features | Preserve the old model's 18-feature order and historical contextual density formula. Empty/single-track cases get finite values. Changing the formula would require retraining. |
| Dropout | Each eligible track independently draws a probability from `[0.1, 0.5]` and a Bernoulli removal. This replaces the old fixed-size subset sampling. Drums have an additional 0.5 removal rule. |
| Protected tracks | Melody plus configured families. All piano/guitar programs are conservatively protected because the paper does not define a guitar-accompaniment classifier. Vocal defaults are GM 52–54 or vocal/voice/singing/choir word tokens in names; substrings such as "sing" in "single violin" do not count. |
| Re-instrumentation | Keep with probability 0.2; otherwise choose the same GM family with conditional probability 0.7, or another musical family whose implementation range covers the track's 10th–90th pitch percentiles. Percussive/sound-effect families are excluded from cross-family choices. Only available bank-0 presets are sampled. Family ranges are explicit constants in `midi.py`; they are engineering choices, not recovered historical settings. |
| Chord operations | A block chord needs overlapping notes with clustered onsets and offsets. Inversion applies to 3–4 distinct tones; arpeggiation applies to 2 or more regular-duration distinct tones (whole through sixteenth notes, including eighths produced by splitting). Both exclude melody, bass and drums. Invert the actual lowest/highest pitch by an octave when MIDI range permits. Arpeggiation uses non-overlapping equal subdivisions over the original duration. The final chord is processed. |
| Splitting | Eligible quarter/half/whole notes are halved in MIDI beat/tick space, retaining pitch, velocity and total duration even across tempo changes. Rounding must leave both fragments at least one MIDI tick. Melody can be split but never chord-inverted or arpeggiated. Drums receive no note edits. Musical-duration tolerance is capped at 0.1 beat to avoid splitting short ornaments at fast tempi. |
| Probability scope | Note probabilities are independently sampled per track and version, rather than shared over the piece. Counts aggregate every track. |
| Audio tempo | The default `tempo_mode: stretch` preserves pitch, so the separately sampled integer transposition remains the intended pitch shift. Explicit `speed` retains legacy resampling: it also changes pitch by `12*log2(factor)` semitones. This choice is recorded. |
| Timing | Whole-waveform pitch shift precedes circular shift and segmentation; speed/stretch happen after segmentation, following Fig. 1. Moving tempo processing to the full song would change positive-pair alignment and is a different protocol. |

Default audio ranges are integer pitch shifts in `[-4, 4]`, sample-accurate
circular shifts within `[-3, 3]` seconds, and tempo factors in `[0.9, 1.1]`.
Only whole windows are retained with `tail_policy: drop`; a source shorter than
one window is always padded to one window. `pad` also retains a partial final
window. Tempo changes make final WAV durations variable, as in the legacy path.

To override melody selection, set `melody_overrides` to a JSON file:

```json
{"train/Track00001": [0, 3]}
```

Indices refer to original MIDI instrument order. The selected tracks must contain
pitched notes. Unknown source IDs, null/empty selections, duplicate indices and
invalid index types fail explicitly; an invalid override never silently falls
back to the classifier. Valid entries for unselected sources remain allowed when
using `--limit`. Unlisted sources continue to use the classifier. The bundled
pickle is trusted repository code; use only trusted replacement pickle models.

## Output, replay and failure recovery

```text
output_root/
  dataset_manifest.json
  train/Track00001/
    transformations.json
    original.mid
    version_0.mid
    version_1.mid
    version_2.mid
    original/0.wav
    original/1.wav
    version_0/0.wav
    ...
```

Each track manifest contains exact sampled audio parameters, MIDI edit plans,
track roles, melody probabilities, source/model/SoundFont hashes, package versions,
implementation hash, per-file checksums and sample counts. Every clip records its
source intervals, including wrap-around, padding, and the output-to-source time
scale. These intervals explain the approximate same-index positives; they do not
claim sample-exact correspondence across shifted versions.

Random states are derived from the global seed, split/source ID, version and stage.
Worker count and processing order do not affect plans. Rendering starts a fresh
synth with fixed settings for each version. Byte identity is checked in tests in
the same environment; different native FluidSynth builds/hardware are not promised
to produce identical PCM. Keep the recorded assets and environment to replay.

```powershell
python -m data.generate_dataset --config path/to/config.yaml --output-root /new/output --replay /old/output/train/Track00001/transformations.json
```

Replay requires matching inputs, processing configuration, implementation and
dependencies. It checks plan hashes, re-identifies melody on the original MIDI,
and independently enforces the actual protected-track roles while applying saved
edits. A checksum-valid edited plan cannot remove or rewrite melody notes. Model
cache entries are keyed by model contents, so replacing a model at the same path
does not silently reuse the old classifier. A normal rerun skips a
track only after verifying its fingerprint, all checksums and audio headers.
Corrupt output with the same fingerprint is rebuilt. Changed parameters/assets
require a new output directory or explicit `--overwrite`. Unrecognized directories
are never replaced. Generation uses a hidden staging directory and an atomic
directory replacement, so an incomplete track is not presented as complete.
An operating-system lock prevents simultaneous generators from writing the same
output root. Split checks also examine existing output from earlier partial runs.
The top-level manifest records the selected sources and any failures; a failed
source causes a nonzero CLI exit code.

## Training boundary

The output structure is compatible with existing MERT extraction. Use the
model-training environment for `python -m data.extract_mert ...` and `train.py`.
Training now reads the supplied train and validation directories directly and
never folds test sources into training. Shared source IDs across feature splits
are rejected. The triplet sampler can select the final version and final time
frame, including a piece with exactly one frame.

## Validation

```powershell
uv pip install --python .venv-augmentation/Scripts/python.exe pytest
.venv-augmentation/Scripts/python.exe -m pytest data/augmentation/tests -q --basetemp=.cache/pytest
```

Two additional model-loader boundary tests run when torch and torchaudio are
installed. They were also validated using the official CPU wheels:

```powershell
uv pip install --python .venv-augmentation/Scripts/python.exe torch==2.6.0 torchaudio==2.6.0 --index https://download.pytorch.org/whl/cpu
```

Integration tests render actual MIDI with a real SoundFont, exercise source audio
resampling, short-piece padding, corruption repair, replay, and verify identical
WAV hashes for serial and parallel generation. Unit tests cover feature schema
compatibility, protected tracks, melody preservation, final chords, pitch-ranked
inversion, and the frequency/duration semantics of both tempo modes.

The release has one supported generation entry, `python -m data.generate_dataset`.
Obsolete `swAIper-no-swiping` generators, duplicate training/MERT scripts, notebooks
and environment exports are excluded. The required historical classifier is kept
in `data/augmentation/assets/`, with its origin and SHA-256 recorded alongside it.

## Paper audit and remaining differences

The review uses the supplied camera-ready paper's Section III and Fig. 1.
V2 matches the stated default probability ranges, instrument keep/conditional
same-family probabilities, quarter/half/whole-note eligibility, melody-preserving
note splitting, 3–4-tone inversion, regular-chord arpeggiation, integer pitch range,
pitch-preserving tempo factors, three variants, and source-piece split isolation.
Note stages and audio stages follow Fig. 1's listed order.

The following are differences or unresolved details; none is presented as an
exact recovery of the historical dataset protocol:

| Item | Current implementation | Difference / uncertainty |
| --- | --- | --- |
| Data and SoundFont | Slakh template points to a user-supplied Musyng file. Functional tests use original toy MIDI and bundled TimGM6mb. | Original Slakh/Musyng assets are lost, so final-paper timbre, data selection, and listening-study quality have not been validated. |
| Melody identification | Bundled historical 18-feature classifier; automatically selects one pitched track. Multiple stems require a manual override. | The paper says "melody tracks" but supplies no selection threshold or weight hash. Selecting the right track in toy tests does not establish recall of every true melody stem on Slakh. Protection is guaranteed for selected stems; classifier mistakes remain possible. |
| Protected roles | All piano, bass and guitar GM families, selected melody, and heuristic vocal roles survive dropout. | Guitar protection is broader than "guitar-accompaniment". Vocal roles use programs/name tokens, not a separately validated role classifier. |
| Instrument ensembles | GM families and explicit pitch-range heuristics; unavailable choices fall back to another available family or keep the original. | The paper does not define its ensemble/register map. These defaults and missing-preset fallback can differ from the historical selection distribution. |
| Note interpretation | Block onsets/offsets use the configured 0.05-second tolerance. Duration tolerance is capped at 0.1 beat. Arpeggio eligibility uses whole through sixteenth notes; its tones partition the original chord span without overlap. Bass/drums receive extra note-edit restrictions; melody may only be split. | The paper does not specify these tolerances, the exact regular-duration grid, arpeggio note releases, or all exclusions. They are explicit implementation choices supporting melody protection. |
| Audio shift and conditioning | Circular shift; canonical-length padding/trimming; mono conversion, fixed synth tail, and recorded peak scaling. | The paper gives a shift range, but no padding/wrap rule, channel policy, tail settings or level-processing recipe. Original mode can read the Slakh mix or resynthesize it; the paper does not state exactly how its original version was rendered. |
| Clip/tempo order | Follow Fig. 1: pitch shift, time shift, segmentation, tempo change. Ten-second source windows become approximately 9.09–11.11-second outputs. Same-index positive windows have recorded source intervals, not sample-exact alignment. | Section III's prose places segmentation last, while Fig. 1 lists tempo change after segmentation. This ambiguity is unresolved. The implementation follows the figure; fixed-duration output would need a separately defined alignment protocol. |
| Dataset population | Preserve supplied split directories; reject duplicate IDs/exact MIDI hashes across splits. | Does not reconstruct the paper's 1,568-source selection, 78 test-source list, or historical seed. Semantic MIDI duplicates with different encodings are not detected. |
| Optional legacy mode | Explicit `tempo_mode: speed` retains pitch-coupled resampling. The default is now `stretch`. | Selecting `speed` adds a noninteger pitch offset and is a documented departure from the independent integer-transposition interpretation. |

Review fixes in v2 include offset-aware block-chord detection, separate inversion
and arpeggiation cardinality rules, split-before-chord processing, musical-time
splitting across tempo changes, whole-waveform pitch shifting, pitch-preserving
default tempo processing, complete padded tail windows, token-based vocal names,
model-content cache invalidation, and checksummed plans with independent replay
melody guards. Regression tests include multiple protected melody stems and
polyphonic melody overrides.
