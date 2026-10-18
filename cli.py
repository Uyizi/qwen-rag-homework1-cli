import argparse
import json
import sys

from rag import Engine, ROOT

HELP = """直接输入问题并回车。
  /method bm25       切换为关键词检索
  /method vector     切换为 E5 向量检索
  /sources [1|2|3]   查看上一次检索原文；不填编号显示全部
  /info              查看模型与知识库信息
  /reindex           更新 corpus 文件后重建索引
  /eval              运行本人填写的正式 20 题，两种方法各一次
  /help              显示命令
  /exit              退出并释放模型
"""


def show_answer(result):
    labels = {"answered": "Qwen", "refused": "Qwen · 资料不足", "rejected": "Qwen · 回答检查未通过"}
    print(f"\n[{labels[result['status']]}]\n{result['answer']}")
    if result["reason"]:
        print("说明：" + result["reason"])
    for index, citation in enumerate(result["citations"], 1):
        print(f"\n出处 {index}：{citation['id']} · {citation['location']}")
        print("原文：" + citation["quote"])
    print(f"\n耗时 {result['latency_seconds']:.2f} 秒，检索 {result['retrieval_seconds']:.3f} 秒。"
          "输入 /sources 查看检索片段。\n")


def show_sources(result, argument=""):
    if result is None:
        print("还没有查询记录。\n")
        return
    hits = result["hits"]
    if argument:
        try:
            index = int(argument)
        except ValueError:
            raise ValueError("用法：/sources，或 /sources 1") from None
        if not 1 <= index <= len(hits):
            raise ValueError(f"片段编号应在 1–{len(hits)} 之间")
        selected = [(index, hits[index - 1])]
    else:
        selected = list(enumerate(hits, 1))
    if not selected:
        print("上一次没有检索到片段。\n")
    for index, hit in selected:
        print(f"\n[{index}] {hit['id']} · {hit['location']} · 得分 {hit['score']:.3f}\n{hit['text']}\n")


def interact(engine, method):
    print("\n知据 · 本地 Qwen 知识库问答")
    print(f"已加载 {engine.stats['documents']} 份资料、{engine.stats['chunks']} 个片段；"
          f"当前检索：{method}。输入 /help 查看命令。\n")
    last = None
    while True:
        try:
            text = input(f"你 [{method}] > ").strip()
        except EOFError:
            break
        if not text:
            continue
        try:
            if not text.startswith("/"):
                print("正在检索并由本地 Qwen 阅读证据……", flush=True)
                last = engine.ask(text, method)
                show_answer(last)
                continue
            command, _, argument = text.partition(" ")
            argument = argument.strip()
            if command in {"/exit", "/quit"}:
                break
            if command == "/help":
                print(HELP)
            elif command == "/method":
                if argument not in {"bm25", "vector"}:
                    raise ValueError("用法：/method bm25，或 /method vector")
                method = argument
                print(f"已切换为 {method}。\n")
            elif command == "/sources":
                show_sources(last, argument)
            elif command == "/info":
                print(f"生成模型：{engine.metadata['model']}\n"
                      f"向量模型：{engine.metadata['embedding_model']}\n"
                      f"当前问答文档：{engine.cfg.get('default_document') or '全部资料'}\n"
                      f"知识库：{engine.stats['documents']} 份资料，{engine.stats['chinese_characters']} 个汉字，"
                      f"{engine.stats['chunks']} 个片段\n资料目录：{ROOT / 'corpus'}\n")
            elif command == "/reindex":
                engine.index(lambda message: print(message, flush=True))
            elif command == "/eval":
                from evaluate import read_csv, run_with_engine, validate_questions
                questions = validate_questions(read_csv(ROOT / "eval" / "questions.csv"))
                run_with_engine(engine, questions)
            else:
                print("未知命令，输入 /help 查看用法。\n")
        except ValueError as exc:
            print(f"输入有误：{exc}\n")
        except (RuntimeError, TimeoutError, OSError) as exc:
            print(f"运行失败：{exc}\n")
            if engine.worker.process.poll() is not None:
                break


def main():
    parser = argparse.ArgumentParser(description="本地 Qwen 知识库问答（交互式 CLI）")
    parser.add_argument("--method", choices=["bm25", "vector"], default="bm25")
    parser.add_argument("--question", help="只回答一个问题，然后退出")
    parser.add_argument("--json", action="store_true", help="配合 --question 输出完整 JSON")
    args = parser.parse_args()
    if args.json and not args.question:
        parser.error("--json 需要配合 --question")
    engine = None
    try:
        engine = Engine(lambda message: print(message, file=sys.stderr, flush=True))
        if args.question:
            result = engine.ask(args.question, args.method)
            print(json.dumps(result, ensure_ascii=False, indent=2)) if args.json else show_answer(result)
        else:
            interact(engine, args.method)
    except KeyboardInterrupt:
        print("\n正在退出……", file=sys.stderr)
    except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
        print(f"启动或运行失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if engine:
            engine.worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
