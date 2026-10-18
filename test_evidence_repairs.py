import unittest
from decimal import Decimal

from pdf_structure import calculation_notes, numeric_value, table_cells
from evidence_checks import (figure_statistics_guard, monotonicity_notes, check_numeric_and_direction,
                             selected_table_facts)


def sample_table():
    return {"id": "sample::tI", "kind": "table", "text": "Table I Accuracy 70.1 72.4", "table": {
        "reference": "Table I", "caption": "Accuracy (%)", "headers": ["Model", "Dataset"],
        "rows": [{"id": "r1", "section": "", "cells": ["Original", "70.1 ± 0.4"]},
                 {"id": "r2", "section": "", "cells": ["Changed", "72.4 ± 0.3"]}]}}


class EvidenceRepairTests(unittest.TestCase):
    def test_decimal_subtraction_uses_real_cells(self):
        notes = calculation_notes([sample_table()], "提高多少")
        self.assertEqual(notes[0]["difference_percentage_points"], "2.3")
        self.assertEqual(notes[0]["left_cell"], "r2:c1")
        self.assertIsNone(numeric_value("This might be 72.4"))

    def test_figure_requires_an_explicit_statistical_table_link(self):
        table = sample_table()
        self.assertEqual(figure_statistics_guard("Figure 2的均值±标准差", [table])["status"], "refused")
        body = {"text": "We evaluate accuracy (Table I, Figure 2)."}
        self.assertIsNone(figure_statistics_guard("Figure 2的均值±标准差", [table, body]))

    def test_direction_reversal_is_rejected_without_hardcoded_parameter(self):
        units = [{"id": "p", "text": "a-b ≥ θ"}]
        constraints = monotonicity_notes(units, "提高θ如何影响筛选")
        self.assertEqual(constraints[0]["increasing_threshold"], "stricter")
        self.assertIsNotNone(check_numeric_and_direction("θ越大，越可能保留更多候选。", units, [], constraints))
        self.assertIsNone(check_numeric_and_direction("θ越大，筛选越严格。", units, [], constraints))

    def test_unsubstantiated_decimal_is_rejected(self):
        units = [{"id": "p", "text": "The observed value was 72.4."}]
        self.assertIsNotNone(check_numeric_and_direction("结果为75.6%。", units, [], []))
        self.assertIsNone(check_numeric_and_direction("结果为72.4%。", units, [], []))

    def test_cells_must_exist_and_show_real_std(self):
        import json
        raw = json.dumps({"facts": [{"id": "sample::tI", "cell": "r2:c1"}]})
        facts, error = selected_table_facts(raw, [sample_table()], "Dataset的准确率")
        self.assertIsNone(error)
        self.assertEqual(facts[0]["value"], "72.4 ± 0.3")
        _, error = selected_table_facts(raw.replace("r2:c1", "r9:c1"), [sample_table()], "Dataset的准确率")
        self.assertIsNotNone(error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
