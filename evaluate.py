import argparse
import csv
import json
import statistics
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_questions(rows):
    if len(rows) != 20 or Counter(r["type"] for r in rows) != {"answerable": 10, "unanswerable": 5, "multihop": 5}:
        raise ValueError("必须有 20 题：answerable 10、unanswerable 5、multihop 5")
    if len({r["id"] for r in rows}) != 20 or any(not r["id"].strip() for r in rows):
        raise ValueError("题号必须非空且唯一")
    if len({r["question"].strip() for r in rows}) != 20:
        raise ValueError("请先由本人填写 20 道不同的题目；禁止用 AI 开发样例冒充正式评测题")
    pending = [r["id"] for r in rows if "review_status" in r and r["review_status"] != "approved"]
    if pending:
        raise ValueError("这份题表是待人工审核的草稿。确认题目、标准答案和出处后，将 review_status 改为 approved。待审：" + "、".join(pending))
    valid_ids = {c["id"] for c in json.loads((ROOT / "artifacts" / "chunks.json").read_text(encoding="utf-8"))}
    for r in rows:
        if not r["question"].strip() or not r["gold_answer"].strip():
            raise ValueError(r["id"] + " 缺少问题或标准答案")
        ids = set(filter(None, r["gold_chunk_ids"].split("|")))
        if r["type"] != "unanswerable" and (not r["gold_sources"].strip() or not ids):
            raise ValueError(r["id"] + " 需要标准答案出处和正确片段 ID")
        if ids - valid_ids:
            raise ValueError(r["id"] + " 有不存在的 gold_chunk_ids")
        alternatives = evidence_sets(r)
        if any(set(group) - valid_ids for group in alternatives):
            raise ValueError(r["id"] + " 有不存在的等价标准证据编号")
        if r["type"] == "multihop" and len(ids) < 2:
            raise ValueError(r["id"] + " 跨片段题至少标注两个必要片段")
    return rows


def evidence_sets(question):
    if question.get("gold_evidence_sets"):
        return json.loads(question["gold_evidence_sets"])
    ids = list(filter(None, question["gold_chunk_ids"].split("|")))
    return [ids] if question["type"] == "multihop" else [[i] for i in ids]


def run_with_engine(engine, questions):

    validate_questions(questions)
    folder = ROOT / "eval" / datetime.now().strftime("run-%Y%m%d-%H%M%S")
    folder.mkdir()
    (folder / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    status = {"ready": True, "transport": "direct local engine", "corpus": engine.stats,
              "environment": engine.metadata, "configuration": engine.cfg,
              "evaluation_role": "post-fix retest on the same approved questions"}
    (folder / "environment.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "implementation").mkdir()
    for name in ["rag.py", "model_worker.py", "evaluate.py", "config.json", "pdf_structure.py", "evidence_checks.py"]:
        shutil.copy2(ROOT / name, folder / "implementation" / name)
    review = []
    # 交替执行两种检索
    with (folder / "predictions.jsonl").open("w", encoding="utf-8") as output:
        for index, q in enumerate(questions):
            methods = ["bm25", "vector"] if index % 2 == 0 else ["vector", "bm25"]
            for method in methods:
                # 标准答案不传入模型
                prediction = engine.ask(q["question"], method)
                prediction.update(id=q["id"], type=q["type"])
                output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                output.flush()
                alternatives = evidence_sets(q)
                gold = set().union(*(set(group) for group in alternatives)) if alternatives else set()
                retrieved = {h["id"] for h in prediction["hits"]}
                context = retrieved | {h["id"] for h in prediction.get("context_hits", [])}
                review.append({"id": q["id"], "method": method, "type": q["type"],
                               "question": q["question"], "gold_answer": q["gold_answer"],
                               "answer": prediction["answer"], "status": prediction["status"],
                               "is_correct": "", "hallucinated": "", "notes": "",
                               "auto_hit_any": int(bool(gold & retrieved)) if gold else "",
                               "auto_hit_all": int(any(set(group) <= retrieved for group in alternatives)) if gold else "",
                               "context_hit_all": int(any(set(group) <= context for group in alternatives)) if gold else "",
                               "latency_seconds": prediction["latency_seconds"]})
                print(q["id"], method, prediction["status"], prediction["latency_seconds"], flush=True)
    write_csv(folder / "human_review.csv", review, list(review[0]))
    (ROOT / "eval" / "latest.txt").write_text(folder.name, encoding="utf-8")
    print("已生成 40 个实际回答。请人工填写", folder / "human_review.csv")
    return folder


def run(args):
    questions = validate_questions(read_csv(ROOT / "eval" / "questions.csv"))
    from rag import Engine
    engine = Engine(lambda message: print(message, flush=True))
    try:
        return run_with_engine(engine, questions)
    finally:
        engine.worker.close()


def summarize(args):
    folder = ROOT / "eval" / (args.run or (ROOT / "eval" / "latest.txt").read_text(encoding="utf-8").strip())
    rows = read_csv(folder / "human_review.csv")
    if len(rows) != 40 or len({(r["id"], r["method"]) for r in rows}) != 40:
        raise ValueError("人工评分表必须保留完整 40 行")
    for row in rows:
        if row["is_correct"] not in {"0", "1"}:
            raise ValueError(row["id"] + " / " + row["method"] + " 缺少人工正确性评分 0/1")
        if row["type"] == "unanswerable" and row["hallucinated"] not in {"0", "1"}:
            raise ValueError("陷阱题必须人工填写 hallucinated=0/1")
        if row["status"] == "rejected" and row["is_correct"] == "1":
            raise ValueError("格式或引用检查失败不能记为正确拒答；请记 0，并在 notes 说明")
    summary = {}
    provenance_path = folder / "scoring-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else None
    for method in ["bm25", "vector"]:
        group = [r for r in rows if r["method"] == method]
        traps = [r for r in group if r["type"] == "unanswerable"]
        answerable = [r for r in group if r["type"] != "unanswerable"]
        correct = sum(int(r["is_correct"]) for r in group)
        hallucinations = sum(int(r["hallucinated"]) for r in traps)
        hits = sum(int(r["auto_hit_any"]) for r in answerable)
        summary[method] = {"correct": correct, "n": 20, "accuracy": correct / 20,
                           "hallucinations": hallucinations, "trap_n": 5, "hallucination_rate": hallucinations / 5,
                           "hit_at_3": hits, "answerable_n": 15, "retrieval_hit_rate": hits / 15,
                           "all_evidence_hit_rate": sum(int(r["auto_hit_all"]) for r in answerable) / 15,
                           "mean_first_complete_answer_seconds": statistics.mean(float(r["latency_seconds"]) for r in group),
                           "by_type": {t: {"correct": sum(int(r["is_correct"]) for r in group if r["type"] == t),
                                          "n": sum(r["type"] == t for r in group)}
                                       for t in ["answerable", "unanswerable", "multihop"]}}
        if provenance:
            summary[method]["scoring_source"] = provenance["source"]
            summary[method]["scoring_provenance"] = provenance_path.name
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "summarize"])
    parser.add_argument("--run")
    arguments = parser.parse_args()
    try:
        (run if arguments.command == "run" else summarize)(arguments)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(1, str(exc) + "\n")
