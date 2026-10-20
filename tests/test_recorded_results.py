"""Recorded-data checks retain failed and undefined outcomes without rerunning solvers."""
from copy import deepcopy
import csv
from pathlib import Path
import unittest

import numpy as np

from experiments.verify_results import (
    EXPECTED_ROWS, equivalent, json_file, same, select, support_rate,
    verify_arrays, verify_distribution, verify_paper,
    verify_public_metadata, verify_stability_rows, verify_summary,
    verify_timing_quantiles, verify_trial_identities,
)

ROOT = Path(__file__).resolve().parents[1]


class SummaryIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"method": "method", "trial": str(index), "support_exact": support,
             "failed": failed, "error": error}
            for index, (support, failed, error) in enumerate((
                ("1", "0", "1"), ("0", "1", "5"), ("", "0", "")))
        ]
        self.summary = {"rows": 3, "aggregate": [{
            "method": "method", "trials": 3, "failures": 1,
            "support_applicable_trials": 2, "support_success_rate": .5,
            "error_available": 2, "error_median": 3.,
            "error_q10_q90": [1.4, 4.6],
        }]}

    def test_undefined_support_is_excluded_rather_than_zero(self):
        verify_summary(self.rows, self.summary, "fixture")
        self.assertEqual(support_rate(self.rows), .5)
        self.assertIsNone(support_rate([self.rows[-1]]))
        self.assertTrue(same("", None))
        self.assertEqual(len(select(self.rows, support_exact=None)), 1)

    def test_quantile_tampering_is_detected(self):
        changed = deepcopy(self.summary)
        changed["aggregate"][0]["error_q10_q90"][0] = 0
        with self.assertRaisesRegex(ValueError, "error_q10_q90"):
            verify_summary(self.rows, changed, "fixture")

    def test_failed_rows_cannot_be_omitted(self):
        with self.assertRaisesRegex(ValueError, "row count"):
            verify_summary(self.rows[:1] + self.rows[2:], self.summary, "fixture")

    def test_overlapping_summary_conditions_are_rejected(self):
        changed = deepcopy(self.summary)
        changed["aggregate"].append(deepcopy(changed["aggregate"][0]))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            verify_summary(self.rows, changed, "fixture")

    def test_unavailable_and_infinite_metrics_stay_accounted_for(self):
        record = {"count": 4, "nonfinite_count": 2, "median": 2., "mean": 2.}
        verify_distribution([1., 3., np.nan, np.inf], record, "fixture")
        record["nonfinite_count"] = 1
        with self.assertRaisesRegex(ValueError, "nonfinite count"):
            verify_distribution([1., 3., np.nan, np.inf], record, "fixture")
        equivalent(np.nan, None, "undefined")
        with self.assertRaises(ValueError):
            equivalent(0, None, "undefined")

    def test_calendar_and_personal_identifiers_are_rejected(self):
        private = ("/" + "Users/person/project", "name" + "@" + "example.invalid",
                   str(2000) + "-01-02T03:04:05", {"created_utc": "removed"})
        for value in private:
            with self.subTest(value=value), self.assertRaises(ValueError):
                verify_public_metadata(value, "fixture")
        verify_public_metadata({"seed": 20260920, "online_seconds": .01,
                                "url": "https://github.com/author/project"}, "fixture")


class RecordedResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {}
        for suite in EXPECTED_ROWS:
            with (ROOT / "results" / suite / "trials.csv").open(newline="", encoding="utf-8") as handle:
                cls.data[suite] = list(csv.DictReader(handle))

    def test_complete_recorded_summaries_and_identities(self):
        self.assertEqual(sum(EXPECTED_ROWS.values()), 93980)
        for suite, expected in EXPECTED_ROWS.items():
            with self.subTest(suite=suite):
                directory = ROOT / "results" / suite
                self.assertEqual(len(self.data[suite]), expected)
                verify_trial_identities(self.data[suite], suite, json_file(directory / "manifest.json"))
                verify_summary(self.data[suite], json_file(directory / "summary.json"), suite)

    def test_manuscript_values_use_separate_serial_timing_trials(self):
        verify_paper(self.data)
        verify_timing_quantiles(ROOT / "results/timing_80db", self.data["timing_80db"])

    def test_saved_array_pairing_and_bound_formulas(self):
        verify_arrays(ROOT, self.data)
        verify_stability_rows(self.data)

    def test_offgrid_rows_keep_unavailable_supports(self):
        offgrid = [row for row in select(self.data["model_stress"], stress="offgrid")
                   if float(row["value"]) != 0]
        self.assertEqual(len(offgrid), 2000)
        self.assertTrue(all(row["support_exact"] == "" for row in offgrid))
        self.assertIsNone(support_rate(offgrid))


if __name__ == "__main__":
    unittest.main()
