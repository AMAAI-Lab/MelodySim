"""Explicit MIDI plans. All indices refer to the original, sorted MIDI tracks."""

from copy import deepcopy
import math
import re

import numpy as np
import pretty_midi


FAMILY_NAMES = (
    "piano", "chromatic_percussion", "organ", "guitar", "bass", "strings",
    "ensemble", "brass", "reed", "pipe", "synth_lead", "synth_pad",
    "synth_effects", "ethnic", "percussive", "sound_effects",
)
# Engineering defaults for the paper's underspecified 'similarly registered'
# family choice. Compare the track's 10th/90th pitch percentiles with these ranges.
FAMILY_RANGES = (
    (21, 108), (48, 96), (36, 96), (40, 88), (24, 60), (36, 96),
    (36, 96), (40, 88), (48, 96), (60, 108), (36, 96), (36, 96),
    (36, 96), (48, 96), (48, 96), (0, 127),
)


def track_roles(midi, melody_indices, config):
    roles = []
    for i, track in enumerate(midi.instruments):
        family = FAMILY_NAMES[track.program // 8]
        reasons = []
        if i in melody_indices:
            reasons.append("melody")
        if not track.is_drum and family in config.protect_families:
            reasons.append(family)
        words = set(re.findall(r"[a-z]+", track.name.lower()))
        if not track.is_drum and config.protect_vocals and (track.program in (52, 53, 54) or words.intersection({"vocal", "vocals", "voice", "voices", "sing", "singer", "singing", "choir", "choral"})):
            reasons.append("vocal")
        roles.append({"index": i, "name": track.name, "program": track.program, "is_drum": track.is_drum,
                      "family": family, "is_melody": i in melody_indices, "protected_reasons": reasons})
    return roles


def _replacement(track, rng, available, config):
    old = track.program
    if track.is_drum or rng.random() < config.keep_instrument_probability:
        return old, "keep"
    same = [p for p in available if p // 8 == old // 8 and p != old]
    low, high = np.percentile([note.pitch for note in track.notes], [10, 90]) if track.notes else (60, 60)
    registered = [p for p in available if p // 8 < 14 and p // 8 != old // 8 and FAMILY_RANGES[p // 8][0] <= low and high <= FAMILY_RANGES[p // 8][1]]
    use_same = rng.random() < config.same_family_probability
    choices = same if use_same else registered
    reason = "same_family" if use_same else "similar_register"
    if not choices:
        choices = same or registered
        reason = "available_family_fallback"
    return (int(rng.choice(choices)), reason) if choices else (old, "no_alternative_preset")


def _groups(notes, eps):
    groups = []
    for index in sorted(range(len(notes)), key=lambda i: (notes[i].start, notes[i].pitch)):
        if not groups or notes[index].start - notes[groups[-1][0]].start > eps:
            groups.append([])
        groups[-1].append(index)
    return groups


def _regular(note, midi, eps, beat_values=(1, 2, 4)):
    # Integrate over the MIDI tempo map via ticks instead of assuming the tempo
    # at note-on also holds until note-off. Cap tolerance for very fast tempi.
    duration_ticks = int(midi.time_to_tick(note.end)) - int(midi.time_to_tick(note.start))
    times, tempi = midi.get_tempo_changes()
    tempo = float(tempi[max(0, np.searchsorted(times, note.start, side="right") - 1)])
    tolerance = min(0.1, eps * tempo / 60) * midi.resolution
    return any(abs(duration_ticks - beats * midi.resolution) <= tolerance for beats in beat_values)


def _split_point(note, midi):
    start, end = int(midi.time_to_tick(note.start)), int(midi.time_to_tick(note.end))
    middle = round((start + end) / 2)
    if not start < middle < end:
        return None
    time = float(midi.tick_to_time(middle))
    return time if note.start < time < note.end else None


def _counts(tracks):
    counts = {kind: sum(op["kind"] == kind for track in tracks for op in track["operations"]) for kind in ("invert", "arpeggiate", "split")}
    counts["removed_tracks"] = sum(track["removed"] for track in tracks)
    counts["arpeggiated_chords"] = sum(len({op["chord_group"] for op in track["operations"] if op["kind"] == "arpeggiate"}) for track in tracks)
    return counts


def sample_midi_plan(midi, melody_indices, rng, available_programs, config):
    roles = track_roles(midi, melody_indices, config)
    tracks = []
    for role, original in zip(roles, midi.instruments):
        remove_probability = float(rng.uniform(*config.remove_track_probability))
        remove = bool(not role["protected_reasons"] and rng.random() < remove_probability)
        drum_remove = bool(original.is_drum and rng.random() < config.remove_drums_probability)
        plan = {**role, "remove_probability": remove_probability, "removed": remove or drum_remove,
                "removed_by_drum_rule": drum_remove, "new_program": original.program,
                "instrument_rule": "removed", "probabilities": {}, "operations": []}
        if plan["removed"] or not original.notes:
            tracks.append(plan)
            continue
        plan["new_program"], plan["instrument_rule"] = _replacement(original, rng, available_programs, config)
        probabilities = {
            "split": float(rng.uniform(*config.note_split_probability)),
            "invert": float(rng.uniform(*config.chord_inversion_probability)),
            "arpeggiate": float(rng.uniform(*config.arpeggiation_probability)),
        }
        plan["probabilities"] = probabilities
        notes, targets = [], []
        # Figure 1: split first, then invert/arpeggiate the resulting block chords.
        # Targets retain original note identity, even after splitting a chord.
        for i, note in enumerate(original.notes):
            midpoint = None
            if not original.is_drum and (not role["is_melody"] or config.split_melody_notes):
                if _regular(note, midi, config.onset_tolerance_seconds) and rng.random() < probabilities["split"]:
                    midpoint = _split_point(note, midi)
            if midpoint is not None:
                plan["operations"].append({"kind": "split", "note_index": i, "at": midpoint})
                fragments = [pretty_midi.Note(note.velocity, note.pitch, note.start, midpoint), pretty_midi.Note(note.velocity, note.pitch, midpoint, note.end)]
            else:
                fragments = [deepcopy(note)]
            for fragment_index, fragment in enumerate(fragments):
                notes.append(fragment)
                targets.append({"note_index": i, "fragment_index": fragment_index})
        # Melody may be split, but never pitch-edited or arpeggiated. This also
        # applies to every manually selected melody track, including polyphony.
        if not original.is_drum and not role["is_melody"] and role["family"] != "bass":
            for group_number, group in enumerate(_groups(notes, config.onset_tolerance_seconds)):
                starts, ends = [notes[i].start for i in group], [notes[i].end for i in group]
                # Similar onsets alone do not make unrelated notes a block chord.
                if len(group) < 2 or len({notes[i].pitch for i in group}) != len(group) or max(starts) >= min(ends) or max(ends) - min(ends) > config.onset_tolerance_seconds:
                    continue
                if len(group) in (3, 4) and rng.random() < probabilities["invert"]:
                    ranked = sorted(group, key=lambda i: notes[i].pitch)
                    choices = [(ranked[0], 12), (ranked[-1], -12)]
                    choices = [(i, delta) for i, delta in choices if 0 <= notes[i].pitch + delta <= 127]
                    if choices:
                        i, delta = choices[int(rng.integers(len(choices)))]
                        notes[i].pitch = int(notes[i].pitch + delta)
                        plan["operations"].append({"kind": "invert", **targets[i], "pitch": notes[i].pitch})
                # The paper's 3-4 tone restriction applies to inversion, not arp.
                if all(_regular(notes[i], midi, config.onset_tolerance_seconds, beat_values=(0.25, 0.5, 1, 2, 4)) for i in group) and rng.random() < probabilities["arpeggiate"]:
                    start, end = min(starts), max(ends)
                    gap = (end - start) / len(group)
                    for rank, i in enumerate(sorted(group, key=lambda i: notes[i].pitch)):
                        notes[i].start, notes[i].end = start + rank * gap, start + (rank + 1) * gap
                        plan["operations"].append({"kind": "arpeggiate", **targets[i], "chord_group": group_number, "start": notes[i].start, "end": notes[i].end})
        tracks.append(plan)
    return {"tracks": tracks, "counts": _counts(tracks)}


def apply_midi_plan(midi, plan, melody_indices=None, config=None):
    result = deepcopy(midi)
    if len(plan["tracks"]) != len(result.instruments):
        raise ValueError("MIDI plan does not match the source track count")
    selected = set(melody_indices if melody_indices is not None else [track["index"] for track in plan["tracks"] if track["is_melody"]])
    expected_roles = track_roles(midi, selected, config) if config is not None else None
    retained = []
    for i, (track, settings) in enumerate(zip(result.instruments, plan["tracks"])):
        if settings["index"] != i or settings["program"] != track.program or settings["is_drum"] != track.is_drum:
            raise ValueError("MIDI plan does not match source track identity")
        role = expected_roles[i] if expected_roles is not None else settings
        if settings["is_melody"] != (i in selected):
            raise ValueError("MIDI plan does not match identified melody tracks")
        if settings["removed"]:
            if i in selected or role["protected_reasons"]:
                raise ValueError(f"MIDI plan removes protected track {i}")
            continue
        track.program = settings["new_program"]
        if isinstance(track.program, bool) or not isinstance(track.program, (int, np.integer)) or not 0 <= track.program <= 127:
            raise ValueError("MIDI plan contains an invalid instrument program")
        track.program = int(track.program)
        fragments = [[note] for note in track.notes]
        for operation in settings["operations"]:
            index = operation["note_index"]
            fragment_index = operation.get("fragment_index", 0)
            if type(index) is not int or not 0 <= index < len(fragments) or type(fragment_index) is not int or not 0 <= fragment_index < len(fragments[index]):
                raise ValueError("MIDI plan contains an invalid note target")
            note = fragments[index][fragment_index]
            kind = operation["kind"]
            if kind != "split" and (i in selected or track.is_drum or FAMILY_NAMES[midi.instruments[i].program // 8] == "bass"):
                raise ValueError(f"MIDI plan changes protected melody/bass/drum notes in track {i}")
            if kind == "invert":
                if type(operation["pitch"]) is not int or not 0 <= operation["pitch"] <= 127 or abs(operation["pitch"] - note.pitch) != 12:
                    raise ValueError("Chord inversion must move a note by one octave")
                note.pitch = operation["pitch"]
            elif kind == "arpeggiate":
                if not all(math.isfinite(operation[key]) for key in ("start", "end")) or not 0 <= operation["start"] < operation["end"]:
                    raise ValueError("Invalid arpeggiation times")
                note.start, note.end = operation["start"], operation["end"]
            elif kind == "split":
                if track.is_drum or (i in selected and config is not None and not config.split_melody_notes):
                    raise ValueError("MIDI plan splits a protected note")
                midpoint = operation["at"]
                expected_midpoint = _split_point(note, midi)
                if len(fragments[index]) != 1 or not _regular(note, midi, config.onset_tolerance_seconds if config else 0.05) or expected_midpoint is None or not math.isfinite(midpoint) or not math.isclose(midpoint, expected_midpoint, abs_tol=1e-9):
                    raise ValueError("Note splitting must halve the original musical duration")
                fragments[index] = [pretty_midi.Note(note.velocity, note.pitch, note.start, midpoint), pretty_midi.Note(note.velocity, note.pitch, midpoint, note.end)]
            else:
                raise ValueError(f"Unknown MIDI operation: {operation['kind']}")
        if i in selected:
            for original, parts in zip(midi.instruments[i].notes, fragments):
                if any(note.pitch != original.pitch or note.velocity != original.velocity for note in parts) or parts[0].start != original.start or parts[-1].end != original.end or any(a.end != b.start for a, b in zip(parts, parts[1:])):
                    raise ValueError(f"MIDI plan does not preserve melody note coverage in track {i}")
        notes = [note for parts in fragments for note in parts]
        track.notes = sorted(notes, key=lambda n: (n.start, n.pitch, n.end))
        retained.append(track)
    result.instruments = retained
    return result
