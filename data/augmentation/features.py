"""The legacy classifier's 18-feature contract, with finite empty-track cases.

Feature order and the historical contextual 'polyphony' formula are intentionally
preserved: changing them requires retraining the classifier, not just refactoring.
"""

import numpy as np


def _mean_std(values):
    return (float(np.mean(values)), float(np.std(values))) if len(values) else (0.0, 0.0)


def _concat(arrays):
    return np.concatenate(arrays) if arrays else np.empty(0)


def get_track_features(midi, fs=20, onset_eps=0.075):
    duration = max(midi.get_end_time(), 1 / fs)
    pitches, velocities, iois = [], [], []
    polyphony, active, densities, durations = [], [], [], []
    for track in midi.instruments:
        notes = sorted(track.notes, key=lambda n: n.start)
        pitches.append(np.array([n.pitch for n in notes]))
        velocities.append(np.array([n.velocity for n in notes]))
        intervals = []
        if len(notes) <= 1:
            intervals = [0.0]
        i = 0
        while i < len(notes):
            j = i + 1
            while j < len(notes) and notes[j].start - notes[j - 1].start < onset_eps:
                j += 1
            if j < len(notes):
                intervals.append(notes[j].start - notes[i].start)
            i = j
        iois.append(np.array(intervals))
        roll = (track.get_piano_roll(fs=fs) > 0).sum(axis=0)
        nonzero = roll[roll > 0]
        polyphony.append(float(nonzero.mean()) if len(nonzero) else 0.0)
        active.append(len(nonzero) / fs)
        densities.append(len(notes) / duration)
        durations.append(active[-1] / duration)
    polyphony, active, densities, durations = map(np.asarray, (polyphony, active, densities, durations))
    features = []
    for i in range(len(midi.instruments)):
        mask = np.arange(len(midi.instruments)) != i
        contextual_polyphony = float((densities[mask] * active[mask]).sum() / active[mask].sum()) if active[mask].sum() else 0.0
        features.append([
            polyphony[i], densities[i], durations[i],
            *_mean_std(pitches[i]), *_mean_std(velocities[i]), *_mean_std(iois[i]),
            contextual_polyphony,
            float(densities[mask].mean()) if mask.any() else 0.0,
            float(durations[mask].mean()) if mask.any() else 0.0,
            *_mean_std(_concat([v for j, v in enumerate(pitches) if j != i])),
            *_mean_std(_concat([v for j, v in enumerate(velocities) if j != i])),
            *_mean_std(_concat([v for j, v in enumerate(iois) if j != i])),
        ])
    return np.asarray(features, dtype=np.float64).reshape(-1, 18)
