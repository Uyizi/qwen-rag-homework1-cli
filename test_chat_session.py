import copy
import json
import queue
import threading
import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import cli
import model_worker
from rag import Worker


class ChatTests(unittest.TestCase):
    def session(self):
        cls = getattr(cli, "ChatSession", None)
        self.assertIsNotNone(cls, "默认聊天会话尚未实现")
        self.requests = []

        def stream(on_text, **request):
            self.requests.append(copy.deepcopy(request))
            on_text("你好")
            return {"text": "你好", "cancelled": False, "output_at_limit": False, "seconds": 1}

        self.engine = SimpleNamespace(
            cfg={"default_document": "paper.pdf"},
            files=[{"file": "paper.pdf"}, {"file": "notes.md"}],
            metadata={"model": "/models/actual-dense"},
            worker=SimpleNamespace(stream=stream),
            ask=lambda *args: self.fail("普通聊天不应检索文档"))
        return cls(self.engine)

    def test_general_chat_bypasses_retrieval_and_remembers_turns(self):
        session = self.session()
        pieces = []
        session.ask("你是谁", pieces.append)
        session.ask("我刚才问了什么", pieces.append)
        self.assertEqual(pieces, ["你好", "你好"])
        self.assertEqual([m["role"] for m in self.requests[1]["messages"]],
                         ["system", "user", "assistant", "user"])
        self.assertIn("actual-dense", self.requests[0]["messages"][0]["content"])
        self.assertNotIn("JSON", self.requests[0]["messages"][0]["content"])
        session.clear()
        self.assertEqual(session.history, [])

    def test_cancelled_or_failed_turn_does_not_enter_history(self):
        session = self.session()
        self.engine.worker.stream = lambda *a, **k: {"text": "半句话", "cancelled": True}
        session.ask("问题", lambda text: None)
        self.assertEqual(session.history, [])
        def fail(*a, **k):
            raise RuntimeError("后端错误")
        self.engine.worker.stream = fail
        with self.assertRaisesRegex(RuntimeError, "后端错误"):
            session.ask("问题", lambda text: None)
        self.assertEqual(session.history, [])

    def test_document_selection_and_rag_do_not_leak_chat_history(self):
        session = self.session()
        session.ask("秘密聊天", lambda text: None)
        session.select_document("2")
        self.assertEqual(self.engine.cfg["default_document"], "notes.md")
        self.assertEqual(session.mode, "rag")
        calls = []
        self.engine.ask = lambda *args: calls.append(args) or {"answer": "文档回答"}
        session.ask("资料问题", lambda text: None)
        self.assertEqual(calls, [("资料问题", "bm25")])
        self.assertEqual(len(session.history), 2)
        session.select_document("all")
        self.assertEqual(self.engine.cfg["default_document"], "")
        with self.assertRaises(ValueError):
            session.select_document("unknown.pdf")

    def test_evaluation_restores_selected_document_even_on_failure(self):
        session = self.session()
        session.select_document("all")
        with self.assertRaises(RuntimeError):
            with session.evaluation_scope():
                self.assertEqual(self.engine.cfg["default_document"], "paper.pdf")
                raise RuntimeError("评测中断")
        self.assertEqual(self.engine.cfg["default_document"], "")

    def test_context_trims_whole_pairs_and_rejects_oversize_question(self):
        trim = getattr(cli, "trim_chat_messages", None)
        self.assertIsNotNone(trim, "上下文裁剪尚未实现")
        messages = [{"role": r, "content": t} for r, t in
                    [("system", "ss"), ("user", "old"), ("assistant", "answer"), ("user", "new")]]
        count = lambda rows: sum(len(m["content"]) for m in rows)
        kept, dropped = trim(messages, count, 6)
        self.assertEqual(kept, [messages[0], messages[-1]])
        self.assertEqual(dropped, 1)
        self.assertEqual(len(messages), 4)
        with self.assertRaises(ValueError):
            trim(messages, count, 4)

    def test_stream_cancel_drains_request_before_next_response(self):
        worker = object.__new__(Worker)
        self.assertTrue(hasattr(worker, "stream"), "流式协议尚未实现")
        worker.lock = threading.Lock()
        worker.process = SimpleNamespace(stdin=StringIO())
        worker.responses = queue.Queue()
        def receive(timeout):
            sent = json.loads(worker.process.stdin.getvalue().splitlines()[0])
            if not getattr(worker, "sent_piece", False):
                worker.sent_piece = True
                return {"event": "text", "request_id": sent["request_id"], "text": "半句"}
            return {"event": "done", "request_id": sent["request_id"], "text": "半句", "cancelled": True}
        worker._receive = receive
        def interrupt(text):
            raise KeyboardInterrupt
        result = worker.stream(interrupt, op="generate", messages=[])
        sent = [json.loads(s) for s in worker.process.stdin.getvalue().splitlines()]
        self.assertTrue(result["cancelled"])
        self.assertEqual(sent[1], {"op": "cancel", "request_id": sent[0]["request_id"]})


class CommandTests(unittest.TestCase):
    def test_modes_and_docs(self):
        reads = iter(["/docs", "/docs all", "/method vector", "资料问题", "/chat", "通用问题", "/clear", "/exit"])
        outputs, calls = [], []
        ui = SimpleNamespace(read=lambda label: next(reads),
                             print=lambda text="", **kwargs: outputs.append(text))
        engine = SimpleNamespace(metadata={"model": "/model/dense"},
                                 cfg={"default_document": "paper.pdf"},
                                 files=[{"file": "paper.pdf"}, {"file": "notes.md"}])

        def answer(session, text, ui):
            calls.append((session.mode, session.method, engine.cfg["default_document"], text))
            return {"answer": "ok"}

        with patch.object(cli, "answer", answer):
            cli.interact(engine, SimpleNamespace(mode="chat", method="bm25"), ui)
        self.assertEqual(calls, [("rag", "vector", "", "资料问题"), ("chat", "vector", "", "通用问题")])
        self.assertTrue(any("notes.md" in text for text in outputs))

    def test_sources_include_additional_evidence(self):
        rows = []
        ui = SimpleNamespace(print=lambda text="", **kwargs: rows.append(text))
        cli.show_sources({"hits": [{"id": "E1", "location": "1", "text": "正文", "score": 1}],
                          "context_hits": [{"id": "E2", "location": "2", "text": "补充证据"}]}, ui)
        self.assertTrue(any("补充证据" in text for text in rows))


class GenerationBoundaryTests(unittest.TestCase):
    def test_chat_uses_tokenizer_turn_boundary_with_exported_eos(self):
        stop_ids = getattr(model_worker, "chat_stop_token_ids", None)
        self.assertIsNotNone(stop_ids, "聊天未校正导出配置中的结束符")
        tokenizer = SimpleNamespace(eos_token_id=248046)
        config = SimpleNamespace(eos_token_id=248044)
        self.assertEqual(stop_ids(tokenizer, config), [248046, 248044])
        config.eos_token_id = [248046, 248044]
        self.assertEqual(stop_ids(tokenizer, config), [248046, 248044])


if __name__ == "__main__":
    unittest.main()
