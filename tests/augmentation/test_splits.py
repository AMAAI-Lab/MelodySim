import pytest

from data.splits import feature_split_directories


def test_training_splits_are_kept_separate_and_test_is_optional(tmp_path):
    for name in ("train", "validation"):
        directory = tmp_path / name
        directory.mkdir()
        for index in range(2):
            (directory / f"{name}{index}.npy").touch()
    directories = feature_split_directories(tmp_path)
    assert directories["train"] == tmp_path / "train"
    (directories["validation"] / "train0.npy").touch()
    with pytest.raises(ValueError, match="both"):
        feature_split_directories(tmp_path)
