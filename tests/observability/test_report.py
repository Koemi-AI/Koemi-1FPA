from __future__ import annotations

import dataclasses
import math
import unittest

from koemi.observability.report import (
    NATS_PER_BIT,
    EpochLearningRate,
    ReportSchemaError,
    RunReport,
    aggregate_run_reports,
    build_run_report,
)


STANDARD_SCHEMA_KEYS = {
    "model",
    "parameters",
    "train_loss_nats",
    "validation_loss_nats",
    "validation_bpb",
    "validation_tokens",
    "train_tokens",
    "train_tokens_per_second",
    "elapsed_seconds",
    "seed",
    "data_seed",
    "epochs",
    "optimizer_steps",
    "batch_size",
    "sequence_length",
    "precision",
    "device",
    "ablation",
    "validation_bpb_std",
    "seeds_used",
    "learning_rate_by_epoch",
    "train_tokens_per_second_excluding_validation",
    "train_tokens_per_second_including_validation",
    "peak_memory_bytes",
    "flops_per_token_estimate",
    "validation_seconds_inside_elapsed",
    "peak_memory_source",
    "parameters_receiving_gradient",
}

STRING_SCHEMA_KEYS = {"model", "precision", "device", "ablation", "peak_memory_source"}


def build_report(**overrides) -> RunReport:
    values = dict(
        model="koemi",
        parameters=76244,
        parameters_receiving_gradient=76196,
        train_loss_nats=2.5,
        validation_loss_nats=2.18,
        validation_tokens=531393,
        train_tokens=10383881,
        elapsed_seconds=120.0,
        validation_seconds_inside_elapsed=0.0,
        seed=17,
        data_seed=0,
        epochs=2,
        optimizer_steps=64,
        batch_size=8,
        sequence_length=96,
        precision="fp32",
        device="cpu",
        ablation="herm",
        learning_rate_by_epoch=(
            EpochLearningRate(epoch=1, learning_rate_start=0.003, learning_rate_end=0.0015),
            EpochLearningRate(epoch=2, learning_rate_start=0.0015, learning_rate_end=0.0),
        ),
        peak_memory_bytes=295940096,
        peak_memory_source="process_peak_resident_set",
    )
    values.update(overrides)
    return build_run_report(**values)


class RunReportSchemaTests(unittest.TestCase):
    def test_report_exposes_exactly_the_standard_schema_keys(self) -> None:
        self.assertEqual(STANDARD_SCHEMA_KEYS, set(build_report().to_dict()))

    def test_model_name_does_not_change_the_schema_keys(self) -> None:
        koemi_keys = set(build_report(model="koemi", ablation="herm").to_dict())
        baseline_keys = set(build_report(model="lstm", ablation="none").to_dict())
        self.assertEqual(koemi_keys, baseline_keys)

    def test_only_the_closed_set_of_fields_carries_text(self) -> None:
        payload = build_report().to_dict()
        text_keys = {key for key, value in payload.items() if isinstance(value, str)}
        self.assertEqual(STRING_SCHEMA_KEYS, text_keys)

    def test_validation_bpb_is_derived_from_nats(self) -> None:
        report = build_report(validation_loss_nats=2.4101358951849194)
        self.assertAlmostEqual(3.477091103851766, report.validation_bpb, places=12)
        self.assertAlmostEqual(report.validation_loss_nats / math.log(2.0), report.validation_bpb, places=12)

    def test_flops_estimate_is_six_times_parameters(self) -> None:
        self.assertEqual(6 * 76244, build_report().flops_per_token_estimate)

    def test_throughput_separates_validation_time_from_training_time(self) -> None:
        report = build_report(elapsed_seconds=100.0, validation_seconds_inside_elapsed=25.0)
        self.assertAlmostEqual(10383881 / 100.0, report.train_tokens_per_second_including_validation, places=9)
        self.assertAlmostEqual(10383881 / 75.0, report.train_tokens_per_second_excluding_validation, places=9)
        self.assertAlmostEqual(report.train_tokens_per_second_including_validation, report.train_tokens_per_second)

    def test_single_seed_report_reports_no_standard_deviation(self) -> None:
        report = build_report()
        self.assertIsNone(report.validation_bpb_std)
        self.assertEqual((17,), report.seeds_used)

    def test_dictionary_round_trip_preserves_the_report(self) -> None:
        report = build_report()
        self.assertEqual(report, RunReport.from_dict(report.to_dict()))

    def test_report_rejects_validation_time_at_or_above_elapsed_time(self) -> None:
        with self.assertRaisesRegex(ReportSchemaError, "smaller than elapsed_seconds"):
            build_report(elapsed_seconds=10.0, validation_seconds_inside_elapsed=10.0)

    def test_report_rejects_an_integer_elapsed_time(self) -> None:
        with self.assertRaisesRegex(ReportSchemaError, "elapsed_seconds must be a float"):
            dataclasses.replace(build_report(), elapsed_seconds=120)

    def test_report_rejects_a_throughput_that_does_not_match_elapsed_time(self) -> None:
        with self.assertRaisesRegex(ReportSchemaError, "including-validation throughput"):
            dataclasses.replace(build_report(), train_tokens_per_second=1.0)

    def test_report_rejects_bits_per_byte_that_are_not_derived_from_nats(self) -> None:
        with self.assertRaisesRegex(ReportSchemaError, "natural log of two"):
            dataclasses.replace(build_report(), validation_bpb=1.0)

    def test_report_rejects_an_unknown_precision(self) -> None:
        with self.assertRaisesRegex(ReportSchemaError, "precision must be one of"):
            build_report(precision="int8")

    def test_report_requires_one_learning_rate_entry_per_epoch(self) -> None:
        with self.assertRaisesRegex(ReportSchemaError, "one entry per epoch"):
            build_report(
                epochs=3,
                learning_rate_by_epoch=(EpochLearningRate(epoch=1, learning_rate_start=0.003, learning_rate_end=0.0),),
            )


class RunReportAggregationTests(unittest.TestCase):
    def build_seed_reports(self) -> list[RunReport]:
        return [
            build_report(seed=17, validation_loss_nats=3.0 * NATS_PER_BIT, elapsed_seconds=100.0),
            build_report(seed=29, validation_loss_nats=3.2 * NATS_PER_BIT, elapsed_seconds=110.0),
            build_report(seed=41, validation_loss_nats=3.4 * NATS_PER_BIT, elapsed_seconds=120.0),
        ]

    def test_aggregate_reports_mean_bits_per_byte_and_seed_standard_deviation(self) -> None:
        aggregate = aggregate_run_reports(self.build_seed_reports())
        self.assertEqual((17, 29, 41), aggregate.seeds_used)
        self.assertAlmostEqual(3.2, aggregate.validation_bpb, places=9)
        self.assertAlmostEqual(0.2, aggregate.validation_bpb_std, places=9)

    def test_aggregate_throughput_uses_mean_elapsed_time(self) -> None:
        aggregate = aggregate_run_reports(self.build_seed_reports())
        self.assertAlmostEqual(110.0, aggregate.elapsed_seconds, places=9)
        self.assertAlmostEqual(10383881 / 110.0, aggregate.train_tokens_per_second, places=6)

    def test_aggregate_rejects_runs_that_measured_different_token_counts(self) -> None:
        reports = self.build_seed_reports()
        reports[1] = build_report(seed=29, validation_tokens=42)
        with self.assertRaisesRegex(ReportSchemaError, "disagree on validation_tokens"):
            aggregate_run_reports(reports)

    def test_aggregate_rejects_runs_from_different_ablations(self) -> None:
        reports = self.build_seed_reports()
        reports[2] = build_report(seed=41, ablation="affine")
        with self.assertRaisesRegex(ReportSchemaError, "disagree on ablation"):
            aggregate_run_reports(reports)

    def test_aggregate_rejects_a_repeated_seed(self) -> None:
        reports = self.build_seed_reports()
        reports[2] = build_report(seed=17)
        with self.assertRaisesRegex(ReportSchemaError, "must not repeat a seed"):
            aggregate_run_reports(reports)

    def test_aggregate_averages_the_learning_rate_of_every_epoch(self) -> None:
        aggregate = aggregate_run_reports(self.build_seed_reports())
        self.assertEqual((1, 2), tuple(schedule.epoch for schedule in aggregate.learning_rate_by_epoch))
        self.assertAlmostEqual(0.003, aggregate.learning_rate_by_epoch[0].learning_rate_start, places=9)


if __name__ == "__main__":
    unittest.main()
