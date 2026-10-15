import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.check_augmentation import validate_soundfont


@pytest.mark.parametrize("content", [b"<html>download failed</html>", b"RIFF\xff\xff\xff\x7fsfbk"])
def test_rejects_download_error_pages_and_truncated_soundfonts(tmp_path, content):
    path = tmp_path / "font.sf2"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="extracted RIFF|truncated"):
        validate_soundfont(path)


@pytest.mark.integration
def test_self_check_exercises_installed_runtime_and_has_failure_exit_code(tmp_path):
    output = tmp_path / "check"
    result = subprocess.run([sys.executable, "-m", "tools.check_augmentation", "--output-dir", str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["melodic_presets"] == 128
    assert report["melody_indices"] == [0]
    assert report["output_samples"] > 0 and Path(report["wav"]).is_file()
    assert "[OK] Pitch shift" in result.stdout
    bad = subprocess.run([sys.executable, "-m", "tools.check_augmentation", "--soundfont", str(tmp_path / "absent.sf2"),
                          "--output-dir", str(tmp_path / "bad")], capture_output=True, text=True)
    assert bad.returncode == 1 and "[FAIL]" in bad.stderr
    assert not (tmp_path / "bad/report.json").exists()
