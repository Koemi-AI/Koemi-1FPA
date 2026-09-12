from __future__ import annotations

import math
import unittest

from benchmarks.run_benchmark import create_parser, evaluation_error, run_single_model
from koemi.observability.report import RunReport


TINY_BUDGET = (
    "--task",
    "bytes",
    "--train-records",
    "4",
    "--evaluation-records",
    "2",
    "--sequence-length",
    "32",
    "--batch-size",
    "2",
    "--epochs",
    "1",
    "--embedding-size",
    "16",
    "--memory-features",
    "4",
    "--local-memory-size",
    "4",
)


def run_model(model_name: str) -> dict:
    arguments = create_parser().parse_args(["--model", model_name, *TINY_BUDGET])
    return run_single_model(arguments)


class BenchmarkMetricTests(unittest.TestCase):
    def test_standard_error_is_defined_for_multiple_token_losses(self) -> None:
        self.assertAlmostEqual(evaluation_error([1.0, 2.0, 3.0]), 1.0 / math.sqrt(3.0), places=7)

    def test_standard_error_is_nan_for_a_single_token(self) -> None:
        self.assertTrue(math.isnan(evaluation_error([1.0])))


class BenchmarkReportSchemaTests(unittest.TestCase):
    def test_weight_seeds_keep_data_token_counts_identical(self) -> None:
        reports = []
        for weight_seed in (17, 29, 41):
            arguments = create_parser().parse_args(
                [
                    "--model", "koemi",
                    "--task", "bytes",
                    "--seed", str(weight_seed),
                    "--data-seed", "123",
                    "--train-records", "4",
                    "--evaluation-records", "2",
                    "--sequence-length", "32",
                    "--batch-size", "2",
                    "--epochs", "1",
                    "--embedding-size", "16",
                    "--memory-features", "4",
                    "--local-memory-size", "4",
                ]
            )
            reports.append(run_single_model(arguments)["report"])
        self.assertEqual(
            {(report["train_tokens"], report["validation_tokens"]) for report in reports},
            {(reports[0]["train_tokens"], reports[0]["validation_tokens"])},
        )

    def test_every_model_emits_the_same_report_schema(self) -> None:
        payloads = {model_name: run_model(model_name) for model_name in ("koemi", "gru", "lstm")}
        schemas = {model_name: set(payload["report"]) for model_name, payload in payloads.items()}
        self.assertEqual(schemas["koemi"], schemas["gru"])
        self.assertEqual(schemas["koemi"], schemas["lstm"])
        for model_name, payload in payloads.items():
            report = RunReport.from_dict(payload["report"])
            self.assertEqual(model_name, report.model)
            self.assertEqual("none" if model_name != "koemi" else "herm", report.ablation)
            self.assertGreater(report.validation_bpb, 0.0)
            self.assertEqual(6 * report.parameters, report.flops_per_token_estimate)

    def test_the_benchmark_keeps_validation_outside_the_training_window(self) -> None:
        report = RunReport.from_dict(run_model("koemi")["report"])
        self.assertEqual(0.0, report.validation_seconds_inside_elapsed)
        self.assertAlmostEqual(
            report.train_tokens_per_second_including_validation,
            report.train_tokens_per_second_excluding_validation,
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
