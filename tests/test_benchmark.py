from __future__ import annotations

import math
import unittest

from benchmarks.run_benchmark import evaluation_error


class BenchmarkMetricTests(unittest.TestCase):
    def test_standard_error_is_defined_for_multiple_token_losses(self) -> None:
        self.assertAlmostEqual(evaluation_error([1.0, 2.0, 3.0]), 1.0 / math.sqrt(3.0), places=7)

    def test_standard_error_is_nan_for_a_single_token(self) -> None:
        self.assertTrue(math.isnan(evaluation_error([1.0])))
