from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from koemi.cli import main, write_utf8


CANONICAL_RECORD = {
    "id": "queue-001",
    "input": "Explain FIFO in one sentence.",
    "thinking": None,
    "output": "FIFO means first in, first out.",
    "metadata": {"source": "test"},
}


class CliOutputTests(unittest.TestCase):
    def test_writes_unicode_output_as_utf8_bytes(self) -> None:
        output = io.BytesIO()
        stdout = type("BufferedStdout", (), {"buffer": output})()
        with patch("koemi.cli.sys.stdout", stdout):
            write_utf8("prefix �")
        self.assertEqual(output.getvalue(), "prefix �\n".encode("utf-8"))


class CliCommandTests(unittest.TestCase):
    def test_trains_and_generates_through_the_router_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            dataset_path = workspace / "dataset.jsonl"
            dataset_path.write_text(json.dumps(CANONICAL_RECORD) + "\n", encoding="utf-8")
            checkpoint_path = workspace / "model.pt"
            train_status = main(
                [
                    "train",
                    "--dataset",
                    str(dataset_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--epochs",
                    "1",
                    "--embedding-size",
                    "16",
                    "--memory-features",
                    "4",
                    "--local-memory-size",
                    "4",
                    "--deep-steps",
                    "1",
                    "--active-specialists",
                    "1",
                    "--routing-mode",
                    "calibration",
                    "--hard-margin",
                    "0.02",
                    "--compute-penalty-weight",
                    "0.1",
                    "--balance-loss-weight",
                    "0.05",
                ]
            )
            self.assertEqual(0, train_status)
            self.assertTrue(checkpoint_path.exists())
            output = io.BytesIO()
            stdout = type("BufferedStdout", (), {"buffer": output})()
            with patch("koemi.cli.sys.stdout", stdout):
                generate_status = main(
                    [
                        "generate",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--prompt",
                        "FIFO",
                        "--max-new-bytes",
                        "4",
                    ]
                )
            self.assertEqual(0, generate_status)
            self.assertTrue(output.getvalue().startswith(b"FIFO"))

    def test_rejects_an_unknown_routing_mode(self) -> None:
        with self.assertRaises(SystemExit):
            main(["train", "--dataset", "missing.jsonl", "--checkpoint", "model.pt", "--routing-mode", "soft"])
