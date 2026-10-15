from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

from data.augmentation.config import MidiConfig
from data.augmentation.features import get_track_features
from data.augmentation.melody import identify_melody
from data.augmentation.midi import apply_midi_plan, sample_midi_plan, track_roles


def test_classifier_feature_contract_matches_legacy(midi):
    # Frozen from the original implementation; no legacy imports are needed.
    path = Path(__file__).with_name("legacy_feature_contract.json")
    np.testing.assert_allclose(get_track_features(midi), json.loads(path.read_text()))


def test_single_empty_and_simultaneous_notes_have_finite_features(midi):
    midi.instruments = [midi.instruments[0], pretty_midi.Instrument(0)]
    assert get_track_features(midi).shape == (2, 18)
    assert np.isfinite(get_track_features(midi)).all()
    midi.instruments = [midi.instruments[0]]
    assert np.isfinite(get_track_features(midi)).all()


def test_actual_classifier_and_override(config, midi):
    prediction = identify_melody(midi, config.classifier_path)
    assert len(prediction["indices"]) == 1
    assert not midi.instruments[prediction["indices"][0]].is_drum
    assert len(prediction["probabilities"]) == len(midi.instruments)
    assert identify_melody(midi, config.classifier_path, [0, 2])["indices"] == [0, 2]


def test_protected_tracks_survive_and_drums_can_drop(midi):
    config = MidiConfig(remove_track_probability=(1, 1), remove_drums_probability=1)
    plan = sample_midi_plan(midi, [0], np.random.default_rng(4), range(128), config)
    assert [p["removed"] for p in plan["tracks"]] == [False, False, False, True, True, True]
    roles = track_roles(midi, [0], config)
    assert roles[1]["family"] == "bass"  # GM program 32, not 33.
    out = apply_midi_plan(midi, plan)
    assert len(out.instruments) == 3


def test_final_chord_and_pitch_rank_regressions(midi):
    # An unsorted, final block chord must be processed, including top/bottom selection.
    chord = pretty_midi.Instrument(48)
    chord.notes = [pretty_midi.Note(70, p, 0, 1) for p in (64, 67, 60)]
    midi.instruments = [chord]
    config = MidiConfig(keep_instrument_probability=1, remove_track_probability=(0, 0),
                        remove_drums_probability=0, note_split_probability=(0, 0),
                        chord_inversion_probability=(1, 1), arpeggiation_probability=(0, 0))
    plan = sample_midi_plan(midi, [], np.random.default_rng(3), range(128), config)
    operation = plan["tracks"][0]["operations"][0]
    assert operation["kind"] == "invert"
    assert operation["note_index"] in (1, 2)
    config.chord_inversion_probability = (0, 0)
    config.arpeggiation_probability = (1, 1)
    result = apply_midi_plan(midi, sample_midi_plan(midi, [], np.random.default_rng(3), range(128), config))
    assert [note.pitch for note in result.instruments[0].notes] == [60, 64, 67]
    assert result.instruments[0].notes[-1].end == 1
    assert all(a.end <= b.start for a, b in zip(result.instruments[0].notes, result.instruments[0].notes[1:]))


def test_melody_splitting_preserves_pitch_and_time(midi):
    original = deepcopy(midi)
    config = MidiConfig(keep_instrument_probability=1, remove_track_probability=(0, 0),
                        remove_drums_probability=0, note_split_probability=(1, 1),
                        chord_inversion_probability=(1, 1), arpeggiation_probability=(1, 1))
    plan = sample_midi_plan(midi, [0], np.random.default_rng(5), range(128), config)
    result = apply_midi_plan(midi, plan)
    melody = result.instruments[0].notes
    assert len(melody) == 2 * len(original.instruments[0].notes)
    for i, note in enumerate(original.instruments[0].notes):
        left, right = melody[2 * i:2 * i + 2]
        assert left.pitch == right.pitch == note.pitch
        assert left.start == note.start and left.end == right.start and right.end == note.end
    assert len(midi.instruments[0].notes) == len(original.instruments[0].notes)
    assert get_track_features(original).shape[1] == 18


def test_all_bass_and_empty_tracks_do_not_crash(midi):
    midi.instruments = [midi.instruments[1], pretty_midi.Instrument(32)]
    plan = sample_midi_plan(midi, [0], np.random.default_rng(9), range(128), MidiConfig())
    assert plan["counts"]["invert"] == 0
    assert apply_midi_plan(midi, plan).instruments


def _forced_notes(**overrides):
    values = dict(keep_instrument_probability=1, remove_track_probability=(0, 0),
                  remove_drums_probability=0, note_split_probability=(0, 0),
                  chord_inversion_probability=(1, 1), arpeggiation_probability=(1, 1))
    values.update(overrides)
    return MidiConfig(**values)


def test_similar_onsets_with_different_offsets_are_not_block_chords():
    midi = pretty_midi.PrettyMIDI()
    track = pretty_midi.Instrument(48)
    track.notes = [pretty_midi.Note(80, pitch, 0, end) for pitch, end in zip((60, 64, 67), (0.2, 0.5, 1.0))]
    midi.instruments = [track]
    plan = sample_midi_plan(midi, [], np.random.default_rng(1), range(128), _forced_notes())
    assert not plan["tracks"][0]["operations"]


@pytest.mark.parametrize("pitches", [(60, 67), (48, 52, 55, 59, 62)])
def test_arpeggiation_is_not_limited_to_three_or_four_tones(pitches):
    midi = pretty_midi.PrettyMIDI()
    track = pretty_midi.Instrument(48)
    track.notes = [pretty_midi.Note(80, pitch, 0, 1) for pitch in pitches]
    midi.instruments = [track]
    plan = sample_midi_plan(midi, [], np.random.default_rng(1), range(128), _forced_notes())
    assert plan["counts"]["invert"] == 0
    assert plan["counts"]["arpeggiated_chords"] == 1
    result = apply_midi_plan(midi, plan)
    assert [n.start for n in result.instruments[0].notes] == pytest.approx([i / len(pitches) for i in range(len(pitches))])


@pytest.mark.parametrize("duration", [0.5, 1.0])
def test_chord_splitting_precedes_chord_operations(duration):
    midi = pretty_midi.PrettyMIDI()
    track = pretty_midi.Instrument(48)
    track.notes = [pretty_midi.Note(80, pitch, 0, duration) for pitch in (60, 64, 67)]
    midi.instruments = [track]
    config = _forced_notes(note_split_probability=(1, 1))
    plan = sample_midi_plan(midi, [], np.random.default_rng(1), range(128), config)
    assert plan["counts"]["split"] == 3
    assert plan["counts"]["arpeggiated_chords"] == 2
    assert [op["kind"] for op in plan["tracks"][0]["operations"][:3]] == ["split"] * 3
    assert len(apply_midi_plan(midi, plan).instruments[0].notes) == 6


def test_splitting_integrates_tempo_changes():
    from io import BytesIO
    import mido
    file = mido.MidiFile(ticks_per_beat=480)
    file.tracks.append(mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000, time=0),
                                      mido.MetaMessage("set_tempo", tempo=1000000, time=480)]))
    file.tracks.append(mido.MidiTrack([mido.Message("program_change", program=73, time=0),
                                      mido.Message("note_on", note=72, velocity=80, time=0),
                                      mido.Message("note_off", note=72, time=1920)]))
    stream = BytesIO()
    file.save(file=stream)
    stream.seek(0)
    midi = pretty_midi.PrettyMIDI(stream)
    plan = sample_midi_plan(midi, [0], np.random.default_rng(1), range(128), _forced_notes(note_split_probability=(1, 1)))
    assert plan["counts"]["split"] == 1
    notes = apply_midi_plan(midi, plan).instruments[0].notes
    assert notes[0].start == 0 and notes[0].end == notes[1].start == 1.5
    assert notes[1].end == 3.5  # Halve four beats, not 3.5 wall-clock seconds.


def test_fast_tempo_does_not_make_short_ornaments_eligible():
    midi = pretty_midi.PrettyMIDI(initial_tempo=1200)
    track = pretty_midi.Instrument(73)
    track.notes = [pretty_midi.Note(80, 72, 0, 0.01)]
    midi.instruments = [track]
    plan = sample_midi_plan(midi, [0], np.random.default_rng(1), range(128), _forced_notes(note_split_probability=(1, 1)))
    assert plan["counts"]["split"] == 0


def test_multiple_melody_tracks_survive_without_note_edits_when_disabled(midi):
    config = _forced_notes(remove_track_probability=(1, 1), remove_drums_probability=1,
                           protect_families=(), note_split_probability=(1, 1), split_melody_notes=False)
    selected = [0, 3]  # Include a polyphonic stem in the melody override.
    plan = sample_midi_plan(midi, selected, np.random.default_rng(1), range(128), config)
    result = apply_midi_plan(midi, plan, melody_indices=selected, config=config)
    assert len(result.instruments) == 2
    for source_index, output in zip(selected, result.instruments):
        assert [(n.pitch, n.start, n.end, n.velocity) for n in output.notes] == [(n.pitch, n.start, n.end, n.velocity) for n in midi.instruments[source_index].notes]


@pytest.mark.parametrize("damage", ["remove", "hide_role", "invert", "arpeggiate"])
def test_replayed_plan_cannot_remove_or_rewrite_melody(midi, damage):
    config = _forced_notes()
    plan = sample_midi_plan(midi, [0], np.random.default_rng(1), range(128), config)
    settings = plan["tracks"][0]
    if damage == "remove":
        settings["removed"] = True
    elif damage == "hide_role":
        settings["is_melody"] = False
    else:
        operation = {"kind": damage, "note_index": 0}
        operation.update(pitch=84) if damage == "invert" else operation.update(start=0.1, end=0.4)
        settings["operations"].append(operation)
    with pytest.raises(ValueError, match="melody|protected"):
        apply_midi_plan(midi, plan, melody_indices=[0], config=config)


def test_vocal_name_matching_does_not_protect_single_violin(midi):
    midi.instruments[0].program = 40
    midi.instruments[0].name = "Single violin"
    midi.instruments[1].program = 40
    midi.instruments[1].name = "lead_vocals"
    roles = track_roles(midi, [], MidiConfig())
    assert not roles[0]["protected_reasons"]
    assert "vocal" in roles[1]["protected_reasons"]


def test_classifier_cache_follows_model_contents(config, tmp_path):
    import pickle
    from data.augmentation.melody import load_classifier
    path = tmp_path / "model.pkl"
    path.write_bytes(Path(config.classifier_path).read_bytes())
    before = load_classifier(str(path))
    updated = deepcopy(before)
    updated.review_marker = "updated"
    path.write_bytes(pickle.dumps(updated))
    after = load_classifier(str(path))
    assert not hasattr(before, "review_marker")
    assert after.review_marker == "updated"
