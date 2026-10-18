import json
import unittest

from evidence_checks import requested_entity_guard, remap_evidence_ids, explicit_context, selected_table_facts, check_numeric_and_direction
from pdf_structure import calculation_notes
from rag import validate_answer


def unit(source, text, section="body"):
    return {"id": source, "file": "demo.pdf", "location": "第1页", "text": text, "section": section}


class FinalEvidenceRepairs(unittest.TestCase):
    def test_negative_difference_magnitude_and_universal_claim(self):
        calculations = [{"difference_percentage_points": "-0.2"}]
        self.assertIsNone(check_numeric_and_direction("略低0.2个百分点", [], calculations, []))
        self.assertIsNotNone(check_numeric_and_direction("两项都优于基线，略低0.2个百分点", [], calculations, []))

    def test_missing_table_source_requires_unique_cited_cell(self):
        table = {"id": "table-one", "kind": "table", "table": {"reference": "Table I", "headers": ["Model", "Demo"],
                 "rows": [{"id": "r1", "section": "", "cells": ["A", "80.2 ± 0.3"]}]}}
        obj = {"citations": [{"id": "table-one"}], "facts": [{"cell": "r1:c1"}]}
        facts, error = selected_table_facts(json.dumps(obj), [table], "Demo")
        self.assertIsNone(error)
        self.assertTrue(facts[0]["source_resolved_from_unique_cited_table"])
        self.assertEqual(facts[0]["value"], "80.2 ± 0.3")
        second = {**table, "id": "table-two"}
        obj["citations"].append({"id": "table-two"})
        self.assertIsNotNone(selected_table_facts(json.dumps(obj), [table, second], "Demo")[1])
        obj["facts"] = [{"id": "unknown", "cell": "r1:c1"}]
        self.assertIsNotNone(selected_table_facts(json.dumps(obj), [table], "Demo")[1])

    def test_requested_dataset_cannot_borrow_real_numbers(self):
        corpus = [unit("c1", "Dataset-A accuracy 87.1"), unit("c2", "Dataset-B", "references")]
        self.assertEqual(requested_entity_guard("在 Dataset-B 上的准确率是多少？", corpus)["status"], "refused")
        self.assertIsNone(requested_entity_guard("在 Dataset-A 上的准确率是多少？", corpus))

    def test_alias_restores_source_and_rejects_unknown(self):
        source = unit("真实论文.pdf::c1", "Grounded fact.")
        obj = {"answerable": True, "answer": "事实", "citations": [{"id": "E1"}], "facts": [{"id": "E1", "cell": "r1:c2"}]}
        remapped = remap_evidence_ids(json.dumps(obj), {"E1": source["id"]})
        self.assertEqual(validate_answer(remapped, [source])["status"], "answered")
        self.assertEqual(json.loads(remapped)["facts"][0]["id"], source["id"])
        obj["citations"][0]["id"] = "E9"
        remapped = remap_evidence_ids(json.dumps(obj), {"E1": source["id"]})
        self.assertEqual(validate_answer(remapped, [source])["status"], "rejected")

    def test_section_and_symbol_context_use_real_document(self):
        corpus = [unit("c1", "III. METHOD\nA. Setup\nThe parameter θ is fixed."),
                  unit("c2", "Additional setup detail."),
                  unit("c3", "B. Cost\nComplexity is O(N)."),
                  unit("c4", "IV. RESULTS\nA. Test\nResults.")]
        self.assertEqual({u["id"] for u in explicit_context(corpus, "结合 III-A 解释")}, {"c1", "c2"})
        self.assertEqual({u["id"] for u in explicit_context(corpus, "θ 参数是什么")}, {"c1"})

    def test_standard_deviation_request_does_not_trigger_comparison(self):
        table = {"id": "t1", "kind": "table", "table": {"caption": "Classification accuracy", "headers": ["Model", "Demo"],
                 "rows": [{"id": "r1", "section": "", "cells": ["ModelA", "80.2 ± 0.3"]},
                          {"id": "r2", "section": "", "cells": ["ModelB", "79.7 ± 0.4"]},
                          {"id": "r3", "section": "", "cells": ["ModelC", "78.5 ± 0.6"]}]}}
        self.assertEqual(calculation_notes([table], "ModelA 在 Demo 上的准确率均值±标准差是多少"), [])
        self.assertEqual(calculation_notes([table], "ModelA 与最强基线在 Demo 的均值差是多少")[0]["difference_percentage_points"], "0.5")


if __name__ == "__main__":
    unittest.main()
