from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from koemi.cli import write_utf8


class CliOutputTests(unittest.TestCase):
    def test_writes_unicode_output_as_utf8_bytes(self) -> None:
        output = io.BytesIO()
        stdout = type("BufferedStdout", (), {"buffer": output})()
        with patch("koemi.cli.sys.stdout", stdout):
            write_utf8("prefix �")
        self.assertEqual(output.getvalue(), "prefix �\n".encode("utf-8"))
