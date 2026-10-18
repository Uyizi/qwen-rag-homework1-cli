import json
import tempfile
import unittest
from pathlib import Path

import fitz

from rag import BM25, Engine, chunk_documents, read_units, validate_answer
import numpy as np


class RegressionTests(unittest.TestCase):
    def test_id_only_citations_render_original_text_and_reject_unknown_id(self):
        source = {"id": "a", "file": "paper.pdf", "location": "1", "text": "Cora\n2,708\nCiteseer\n3,327"}
        obj = {"answerable": True, "answer": "Cora和Citeseer", "missing": [], "citations": [{"id": "a"}]}
        final = validate_answer(json.dumps(obj), [source])
        self.assertEqual(final["status"], "answered")
        self.assertEqual(final["citations"][0]["quote"], source["text"])
        obj["citations"][0]["id"] = "not_retrieved"
        self.assertEqual(validate_answer(json.dumps(obj), [source])["status"], "rejected")

    def test_pdf_whitespace_is_allowed_but_changed_numbers_are_rejected(self):
        source = {"id": "a", "file": "paper.pdf", "location": "第1页",
                  "text": "Dataset accuracy is\n80.2 ± 0.3 percent."}
        obj = {"answerable": True, "answer": "80.2 ± 0.3", "missing": [],
               "citations": [{"id": "a", "quote": "accuracy is 80.2 ± 0.3 percent."}]}
        final = validate_answer(json.dumps(obj), [source])
        self.assertEqual(final["status"], "answered")
        self.assertIn("\n", final["citations"][0]["quote"])
        obj["citations"][0]["quote"] = "accuracy is 80.3 ± 0.3 percent."
        self.assertEqual(validate_answer(json.dumps(obj), [source])["status"], "rejected")

    def test_quote_from_another_chunk_remains_rejected(self):
        raw = json.dumps({"answerable": True, "answer": "80.2", "missing": [],
                          "citations": [{"id": "a", "quote": "accuracy is 80.2 percent."}]})
        hits = [{"id": "a", "text": "accuracy is 77.5 percent.", "file": "p.pdf", "location": "1"},
                {"id": "b", "text": "accuracy is\n80.2 percent.", "file": "p.pdf", "location": "2"}]
        self.assertEqual(validate_answer(raw, hits)["status"], "rejected")

    def test_reference_heading_keeps_original_chunks_but_marks_reference_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.txt"
            path.write_text("This is the original body.\nREFERENCES\n" + "[1] Another paper.\n" * 6, encoding="utf-8")
            chunks, _ = chunk_documents(Path(directory), chunk_chars=35, overlap=5)
            self.assertTrue(any(c["section"] == "references" for c in chunks))
            self.assertNotIn("Another paper", " ".join(c["retrieval_text"] for c in chunks))
            self.assertIn("Another paper", " ".join(c["text"] for c in chunks))

    def test_both_retrievers_exclude_reference_candidates_before_topk(self):
        engine = object.__new__(Engine)
        engine.cfg = {"default_document": "paper.pdf", "exclude_references": True, "top_k": 1}
        engine.chunks = [{"id": "body", "file": "paper.pdf", "section": "body", "text": "graph body"},
                         {"id": "ref", "file": "paper.pdf", "section": "references", "text": "graph graph graph"}]
        engine.bm25 = BM25([c["text"] for c in engine.chunks])
        engine.vectors = np.asarray([[0.5, 0.5], [1.0, 0.0]])
        class WorkerStub:
            def embed(self, texts, query=False):
                return np.asarray([[1.0, 0.0]])
        engine.worker = WorkerStub()
        for method in ["bm25", "vector"]:
            self.assertEqual(engine.retrieve("graph", method)[0]["id"], "body")

    def test_chinese_bm25(self):
        scores = BM25(["监听器接收事件通知。", "向量模型编码文档。"]).scores("监听器如何接收事件")
        self.assertGreater(scores[0], scores[1])

    def test_fabricated_quote_rejected(self):
        raw = json.dumps({"answerable": True, "answer": "推断答案", "missing": [],
                          "citations": [{"id": "a", "quote": "根本不存在的原文"}]})
        self.assertEqual(validate_answer(raw, [{"id": "a", "text": "文档只有这句话。"}])["status"], "rejected")

    def test_malformed_model_fields_do_not_crash(self):
        raw = '{"answerable":false,"missing":7,"citations":[],"answer":"不知道"}'
        self.assertEqual(validate_answer(raw, [])["status"], "rejected")

    def test_unhashable_citation_id_does_not_crash(self):
        raw = '{"answerable":true,"answer":"答案","missing":[],"citations":[{"id":[],"quote":"引用至少六个字"}]}'
        self.assertEqual(validate_answer(raw, [])["status"], "rejected")

    def test_mixed_scanned_pdf_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.pdf"
            doc = fitz.open()
            doc.new_page().insert_text((60, 60), "A searchable text page.")
            page = doc.new_page()
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 30, 30), False)
            pix.clear_with(200)
            page.insert_image(fitz.Rect(40, 40, 200, 200), stream=pix.tobytes("png"))
            doc.save(path)
            doc.close()
            with self.assertRaisesRegex(ValueError, "OCR|扫描"):
                read_units(path)

    def test_three_text_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in [".txt", ".md"]:
                path = Path(directory) / ("document" + suffix)
                path.write_text("这是一份测试资料。", encoding="utf-8")
                self.assertIn("测试资料", read_units(path)[0][1])
            path = Path(directory) / "document.pdf"
            doc = fitz.open()
            doc.new_page().insert_text((60, 60), "PDF extraction works.")
            doc.save(path)
            doc.close()
            self.assertIn("PDF extraction works", read_units(path)[0][1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
