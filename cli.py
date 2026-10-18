import argparse
import json
import sys

from chat_session import ChatSession, trim_chat_messages
from rag import Engine, ROOT
from terminal_ui import TerminalUI


HELP = """直接输入即可连续聊天 Enter 发送 Alt+Enter 换行 Tab 补全命令
  /chat             普通聊天 使用模型知识与对话上下文
  /rag [问题]       文档问答 根据当前资料回答并显示出处
  /docs             列出资料
  /docs all|编号    选择全部资料或某一份 同时进入文档模式
  /method bm25|vector  选择关键词或 E5 检索 同时进入文档模式
  /new 或 /clear    清空聊天上下文
  /sources [编号]   查看上次文档问答的检索原文
  /model 或 /info   查看实际模型与当前模式
  /reindex          导入 corpus 中新增或修改的资料
  /eval             以启动配置中的文档范围运行正式20题
  /help             显示命令
  /exit             退出并释放模型
生成时 Ctrl+C 停止当前回答 输入时 Ctrl+C 清空输入 Ctrl+D 退出
"""


def show_answer(result, ui):
    labels = {"answered": "Qwen · 文档", "refused": "Qwen · 资料不足", "rejected": "Qwen · 校验失败"}
    ui.print(f"\n{labels[result['status']]}\n{result['answer']}")
    if result["reason"]:
        ui.print("说明：" + result["reason"])
    for index, citation in enumerate(result["citations"], 1):
        ui.print(f"出处 {index}：{citation['file']} · {citation['location']}", dim=True)
    ui.print(f"{result['latency_seconds']:.2f} 秒 · /sources 查看检索原文\n", dim=True)


def show_sources(result, ui, argument=""):
    if result is None:
        ui.print("还没有文档查询记录\n")
        return
    hits = list(result["hits"])
    ids = {hit["id"] for hit in hits}
    hits.extend(hit for hit in result.get("context_hits", []) if hit["id"] not in ids)
    if argument:
        if not argument.isdecimal() or not 1 <= int(argument) <= len(hits):
            raise ValueError(f"片段编号应在 1 至 {len(hits)} 之间")
        selected = [(int(argument), hits[int(argument) - 1])]
    else:
        selected = list(enumerate(hits, 1))
    if not selected:
        ui.print("上一次没有检索到片段\n")
    for index, hit in selected:
        score = f"得分 {hit['score']:.3f}" if "score" in hit else "补充上下文"
        ui.print(f"\n[{index}] {hit['id']} · {hit['location']} · {score}\n{hit['text']}\n")


def answer(session, question, ui):
    if session.mode == "rag":
        ui.print("检索资料中…", dim=True)
        result = session.ask(question, lambda text: None)
        show_answer(result, ui)
        return result
    with ui.answer() as on_text:
        result = session.ask(question, on_text)
    if result.get("cancelled"):
        ui.print("已停止 本轮未加入上下文", dim=True)
    elif result.get("output_at_limit"):
        ui.print("达到输出长度上限 本轮未加入上下文 可缩小问题范围", dim=True)
    if result.get("trimmed_turns"):
        ui.print(f"上下文已移除最早 {result['trimmed_turns']} 轮", dim=True)
    return result


def interact(engine, args, ui):
    session = ChatSession(engine, args.mode, args.method)
    ui.print("\n知据 · 本地 Qwen")
    ui.print(f"模型 {engine.metadata['model'].rsplit('/', 1)[-1]}", dim=True)
    ui.print("Enter 发送 · Alt+Enter 换行 · Ctrl+C 停止 · /help 命令", dim=True)
    ui.print("默认普通聊天 · /rag 文档问答 · /docs 选择资料\n", dim=True)
    last = None
    while True:
        label = "聊天" if session.mode == "chat" else f"文档 {session.method}"
        try:
            text = ui.read(label)
        except EOFError:
            break
        except KeyboardInterrupt:
            ui.print()
            continue
        if not text:
            continue
        try:
            if not text.startswith("/"):
                result = answer(session, text, ui)
                if session.mode == "rag":
                    last = result
                continue
            command, _, argument = text.partition(" ")
            argument = argument.strip()
            if command in {"/exit", "/quit"}:
                break
            if command == "/help":
                ui.print(HELP)
            elif command in {"/chat", "/rag"}:
                session.mode = command[1:]
                ui.print("普通聊天" if session.mode == "chat" else
                         "文档问答 · " + (engine.cfg.get("default_document") or "全部资料"), dim=True)
                if argument:
                    result = answer(session, argument, ui)
                    if session.mode == "rag":
                        last = result
            elif command == "/method":
                if argument not in {"bm25", "vector"}:
                    raise ValueError("用法 /method bm25 或 /method vector")
                session.method, session.mode = argument, "rag"
                ui.print(f"文档问答 · {argument}", dim=True)
            elif command == "/docs":
                if argument:
                    ui.print("文档问答 · " + session.select_document(argument), dim=True)
                else:
                    for index, document in enumerate(engine.files, 1):
                        ui.print(f"{index}  {document['file']}")
                    ui.print("当前 " + (engine.cfg.get("default_document") or "全部资料"), dim=True)
                    ui.print("/docs all 检索全部 · /docs 编号 选择一份\n", dim=True)
            elif command in {"/new", "/clear"}:
                session.clear()
                last = None
                ui.print("已清空聊天上下文\n", dim=True)
            elif command == "/sources":
                show_sources(last, ui, argument)
            elif command in {"/info", "/model"}:
                ui.print(f"模型 {engine.metadata['model']}\n"
                         f"向量 {engine.metadata['embedding_model']}\n"
                         f"当前模式 {session.mode} · 检索 {session.method}\n"
                         f"资料范围 {engine.cfg.get('default_document') or '全部资料'}\n"
                         f"知识库 {engine.stats['documents']} 份 · {engine.stats['chunks']} 个片段\n"
                         f"聊天上下文 {len(session.history) // 2} 轮\n")
            elif command == "/reindex":
                engine.index(ui.print)
            elif command == "/eval":
                from evaluate import read_csv, run_with_engine, validate_questions
                questions = validate_questions(read_csv(ROOT / "eval" / "questions.csv"))
                with session.evaluation_scope():
                    run_with_engine(engine, questions)
            else:
                ui.print("未知命令 输入 /help 查看用法\n")
        except ValueError as exc:
            ui.print(f"输入有误：{exc}\n")
        except (RuntimeError, TimeoutError, OSError) as exc:
            ui.print(f"运行失败：{exc}\n")
            if engine.worker.process.poll() is not None:
                break
        except KeyboardInterrupt:
            ui.print("\n文档请求已中断 正在关闭模型 请重新启动")
            break


def main():
    parser = argparse.ArgumentParser(description="本地 Qwen 聊天与知识库 CLI")
    parser.add_argument("--mode", choices=["chat", "rag"], default="chat")
    parser.add_argument("--method", choices=["bm25", "vector"], default="bm25")
    parser.add_argument("--plain", action="store_true", help="使用纯文本界面")
    parser.add_argument("--question", help="只回答一个问题 然后退出")
    parser.add_argument("--json", action="store_true", help="输出单次问答的完整 JSON")
    args = parser.parse_args()
    if args.json and not args.question:
        parser.error("--json 需要配合 --question")
    engine = None
    try:
        engine = Engine(lambda message: print(message, file=sys.stderr, flush=True))
        ui = TerminalUI(args.plain)
        if args.question:
            session = ChatSession(engine, args.mode, args.method)
            if args.json:
                result = session.ask(args.question, lambda text: None)
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                answer(session, args.question, ui)
        else:
            interact(engine, args, ui)
    except KeyboardInterrupt:
        print("\n正在退出…", file=sys.stderr)
    except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
        print(f"启动或运行失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if engine:
            engine.worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
