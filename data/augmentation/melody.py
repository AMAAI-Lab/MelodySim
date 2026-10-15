"""Load the bundled melody classifier once per worker and identify original tracks."""

from functools import lru_cache
import pickle

import numpy as np

from .features import get_track_features
from .slakh import file_sha256


def load_classifier(path):
    return _load_classifier(path, file_sha256(path))


@lru_cache(maxsize=4)
def _load_classifier(path, digest):
    # Only use the bundled model or a trusted user-supplied model. Pickle is code.
    with open(path, "rb") as stream:
        model = pickle.load(stream)
    if model.n_features_in_ != 18:
        raise ValueError("Melody classifier must accept the legacy 18-feature schema")
    return model


def identify_melody(midi, classifier_path, override=None):
    candidates = [i for i, track in enumerate(midi.instruments) if track.notes and not track.is_drum]
    if not candidates:
        raise ValueError("No pitched track is available for melody identification")
    if override is not None:
        if not isinstance(override, list) or not override or any(type(i) is not int or i not in candidates for i in override) or len(set(override)) != len(override):
            raise ValueError("Melody override must contain unique nonempty, pitched track indices")
        return {"method": "override", "indices": sorted(override), "probabilities": None, "feature_schema": "legacy-18-v1"}
    features = get_track_features(midi)
    if not np.isfinite(features).all():
        raise ValueError("Melody features contain nonfinite values")
    model = load_classifier(str(classifier_path))
    positive = list(model.classes_).index(1)
    probabilities = model.predict_proba(features)[:, positive]
    selected = max(candidates, key=lambda i: probabilities[i])
    return {"method": "classifier", "indices": [selected], "probabilities": probabilities.tolist(), "feature_schema": "legacy-18-v1"}
