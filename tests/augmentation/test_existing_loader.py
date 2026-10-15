"""Optional integration with the existing model stack's real CPU data loaders."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Install CPU torch/torchaudio to check the model-stack boundary")
pytest.importorskip("torchaudio")

from data.dataloader import TripletDataset, create_audio_dataloader
from data.augmentation.pipeline import generate_dataset


def test_single_frame_triplets_include_last_version(tmp_path):
    for source in range(2):
        values = np.zeros((2, 1, 4, 3, 768), dtype=np.float32)
        values[0] = 10 * source + 1
        values[1] = 10 * source + 2
        np.save(tmp_path / f"Track{source}.npy", values)
    (tmp_path / "metadata.json").write_text("{}")
    dataset = TripletDataset(str(tmp_path))
    torch.manual_seed(5)
    observed = set()
    for _ in range(20):
        anchor, positive, negative = dataset[0]
        observed.add(int(anchor[0, 0]))
        assert anchor.shape == positive.shape == negative.shape == (3072, 3)
        assert anchor[0, 0] != positive[0, 0]
        assert negative[0, 0] >= 11
    assert observed == {1, 2}


@pytest.mark.integration
def test_generated_audio_loads_in_existing_dataloader(config):
    config.num_versions = 1
    generate_dataset(config)
    loader = create_audio_dataloader(str(Path(config.output_root) / "train"), batch_size=1,
                                     num_workers=0, verbose=False)
    batch = next(iter(loader))
    assert list(batch) == ["TrackA"]
    assert len(batch["TrackA"]) == 2
    assert all(version.ndim == 3 and version.shape[0] == 2 for version in batch["TrackA"])
