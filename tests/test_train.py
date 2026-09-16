import subprocess
import sys
from pathlib import Path


def test_train_script():
    result = subprocess.run(
        [
            sys.executable,
            "src/train.py",
            "--test-size",
            "0.2",
            "--random-state",
            "42",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Accuracy:" in result.stdout

    output_file = Path("outputs/confusion_matrix.png")
    assert output_file.exists()