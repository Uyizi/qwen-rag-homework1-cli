import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import rag
from evidence_checks import explicit_context


def unit(identifier, text, file="sample.pdf", location="第 1 页", chunk=1):
    return {"id": identifier, "file": file, "text": text, "retrieval_text": text,
            "location": location, "chunk": chunk, "section": "body"}


class DocumentMetadataTests(unittest.TestCase):
    def test_metadata_questions_locate_header_without_keyword_overlap(self):
        header = unit("c1", "Adaptive Computing\nMina Park\nExample Institute\nAbstract—A study.")
        body = unit("c2", "The experiment reports accuracy.", location="第 2 页", chunk=2)
        for question in ["这篇文章的作者是谁，来自哪里", "这篇文章的标题", "Who wrote this paper?", "What is the title?"]:
            with self.subTest(question=question):
                found = explicit_context([header, body], question)
                self.assertIn(header, found)
                self.assertNotIn(body, found)
        self.assertEqual(explicit_context([header, body], "训练使用的 GPU 型号是什么"), [])

    def test_long_header_is_read_until_abstract_and_keeps_exact_sources(self):
        first = unit("c1", "Adaptive Computing\nMina Park")
        second = unit("c2", "Example Institute\nAbstract—A study.", chunk=2)
        third = unit("c3", "More experiments.", chunk=3)
        self.assertEqual(explicit_context([first, second, third], "作者和单位是什么"), [first, second])

    def test_title_with_no_bm25_hit_still_reaches_generation_and_citations(self):
        header = unit("c1", "Adaptive Computing\nMina Park\nExample Institute\nAbstract—A study.")
        engine = object.__new__(rag.Engine)
        engine.cfg = {"default_document": "sample.pdf", "max_new_tokens": 128, "neighbor_chunks": 0}
        engine.chunks = [header, unit("other", "Other Person", file="other.pdf")]
        engine.lock = threading.Lock()
        engine.prepare_query = lambda question: {"query": "What is the title?", "status": "translated"}
        engine.retrieve = lambda question, method: []
        calls = []

        def generate(**request):
            calls.append(request)
            evidence = json.loads(request["messages"][1]["content"])["证据"]
            self.assertEqual([e["text"] for e in evidence], [header["text"]])
            return {"text": json.dumps({"answerable": True, "answer": evidence[0]["text"],
                                        "citations": [{"id": evidence[0]["id"]}], "missing": []})}

        engine.worker = SimpleNamespace(call=generate)
        with tempfile.TemporaryDirectory() as folder, patch.object(rag, "ROOT", Path(folder)):
            result = engine.ask("这篇文章的标题", "bm25")
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["citations"][0]["id"], "c1")
        self.assertEqual(result["citations"][0]["quote"], header["text"])

    def test_real_imported_paper_has_header_in_both_retrieval_paths(self):
        chunks = json.loads((rag.ROOT / "artifacts/chunks.json").read_text(encoding="utf-8"))
        engine = object.__new__(rag.Engine)
        engine.cfg = {"default_document": "HAGNN论文.pdf", "neighbor_chunks": 0}
        engine.chunks = chunks
        unrelated = next(c for c in chunks if c["file"] == "HAGNN论文.pdf" and c["chunk"] == 30)
        for hits in [[], [unrelated]]:
            context = engine.expand_evidence(hits, "这篇文章的作者是谁，来自哪里")
            headers = [u for u in context if u["id"] == "HAGNN论文.pdf::c001"]
            self.assertEqual(len(headers), 1)
            self.assertIn("Chenjie Yin", headers[0]["text"])
            self.assertIn("Jiaotong University", headers[0]["text"])


if __name__ == "__main__":
    unittest.main()
