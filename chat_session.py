from contextlib import contextmanager
from pathlib import PurePosixPath


def trim_chat_messages(messages, count_tokens, max_input_tokens):
    if max_input_tokens <= 0:
        raise ValueError("上下文长度必须大于生成长度")
    kept = list(messages)
    dropped = 0
    while count_tokens(kept) > max_input_tokens:
        if len(kept) <= 2:
            raise ValueError("当前问题超过上下文容量 请缩短问题或分段发送")
        kept = [kept[0], *kept[3:]]
        dropped += 1
    return kept, dropped


class ChatSession:
    def __init__(self, engine, mode="chat", method="bm25"):
        self.engine = engine
        self.mode = mode
        self.method = method
        self.history = []
        self.evaluation_document = engine.cfg.get("default_document", "")
        model = PurePosixPath(engine.metadata["model"]).name
        self.system = (
            f"你是本地运行的 Qwen 助手 当前加载的模型目录为 {model}。"
            "默认使用简体中文 直接回答问题 简洁清楚。"
            "可以使用自身知识并结合对话上下文回答 不确定时明确说明。"
            "你没有联网搜索或执行命令的工具 不得声称已经联网或执行操作。"
            "当前是普通聊天 未检索知识库 如需基于资料回答 提示用户使用 /rag。")

    def clear(self):
        self.history.clear()

    def select_document(self, argument):
        files = [item["file"] for item in self.engine.files]
        if argument == "all":
            selected = ""
        elif argument.isdecimal() and 1 <= int(argument) <= len(files):
            selected = files[int(argument) - 1]
        elif argument in files:
            selected = argument
        else:
            raise ValueError("使用 /docs all 或 /docs 资料编号或完整文件名")
        self.engine.cfg["default_document"] = selected
        self.mode = "rag"
        return selected or "全部资料"

    @contextmanager
    def evaluation_scope(self):
        selected = self.engine.cfg.get("default_document", "")
        self.engine.cfg["default_document"] = self.evaluation_document
        try:
            yield
        finally:
            self.engine.cfg["default_document"] = selected

    def ask(self, question, on_text):
        if not question.strip():
            raise ValueError("请输入问题")
        if self.mode == "rag":
            return self.engine.ask(question, self.method)
        user = {"role": "user", "content": question}
        result = self.engine.worker.stream(
            on_text, op="generate", chat=True,
            messages=[{"role": "system", "content": self.system}, *self.history, user],
            max_new_tokens=self.engine.cfg.get("chat_max_new_tokens", 1024),
            context_tokens=self.engine.cfg.get("chat_context_tokens", 8192))
        if result.get("text") and not result.get("cancelled") and not result.get("output_at_limit"):
            self.history = self.history[2 * result.get("trimmed_turns", 0):]
            self.history.extend([user, {"role": "assistant", "content": result["text"]}])
        return result
