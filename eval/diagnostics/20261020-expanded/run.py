import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from rag import Engine


def complete(ids, alternatives):
    return any(set(group) <= ids for group in alternatives)


def main():
    base = Path(__file__).resolve().parent
    data = json.loads((base / "questions.json").read_text(encoding="utf-8"))
    folder = base / datetime.now().strftime("run-%Y%m%d-%H%M%S")
    folder.mkdir()
    shutil.copy2(base / "questions.json", folder / "questions.json")
    engine = None
    try:
        engine = Engine(lambda message: print(message, flush=True))
        known = {c["id"] for c in engine.chunks}
        for question in data["questions"]:
            for group in question["gold_evidence_sets"]:
                if not set(group) <= known:
                    raise ValueError(question["id"] + " has unknown evidence")
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        environment = {"revision": revision, "model": engine.metadata, "configuration": dict(engine.cfg),
                       "purpose": data["purpose"], "semantic_scoring": "pending AI review against source", "attempts_per_question_method": 1}
        (folder / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
        header_index = next(i for i, c in enumerate(engine.chunks) if c["id"] == "AHGNN论文.pdf::c001")
        fresh = engine.worker.embed([engine.chunks[header_index]["retrieval_text"]])[0]
        norms = np.linalg.norm(engine.vectors, axis=1)
        health = {"vector_shape": list(engine.vectors.shape), "norm_min": float(norms.min()), "norm_max": float(norms.max()),
                  "header_fresh_cached_cosine": float(fresh @ engine.vectors[header_index]), "queries": []}
        candidates = [i for i, c in enumerate(engine.chunks) if c["file"] == "AHGNN论文.pdf" and c.get("section") != "references"]
        queries = ["Chenjie Yin Xi’an Jiaotong University", "Who is the author of this article and where are they from?",
                   "这篇文章的作者是谁，来自哪里", "What is the title of this paper?", "contact email author affiliation university"]
        labelled = "Document title, authors, affiliation, university, contact email:\n" + engine.chunks[header_index]["text"].split("Abstract")[0]
        labelled_vector = engine.worker.embed([labelled])[0]
        for query in queries:
            vector = engine.worker.embed([query], query=True)[0]
            scores = engine.vectors @ vector
            ranked = sorted(candidates, key=lambda i: -float(scores[i]))
            shadow = scores.copy()
            shadow[header_index] = float(labelled_vector @ vector)
            ranked_shadow = sorted(candidates, key=lambda i: -float(shadow[i]))
            health["queries"].append({"query": query, "header_rank": ranked.index(header_index) + 1,
                "top3": [engine.chunks[i]["id"] for i in ranked[:3]],
                "labelled_header_shadow_rank": ranked_shadow.index(header_index) + 1})
        (folder / "embedding-health.json").write_text(json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8")
        print("EMBEDDING_HEALTH", json.dumps(health, ensure_ascii=False), flush=True)
        rows = []
        with (folder / "predictions.jsonl").open("w", encoding="utf-8") as raw:
            for index, question in enumerate(data["questions"]):
                engine.cfg["default_document"] = question["document"]
                methods = ["bm25", "vector"] if index % 2 == 0 else ["vector", "bm25"]
                for method in methods:
                    result = engine.ask(question["question"], method)
                    result.update(diagnostic_id=question["id"], diagnostic_category=question["category"])
                    raw.write(json.dumps(result, ensure_ascii=False) + "\n")
                    raw.flush()
                    alternatives = question["gold_evidence_sets"]
                    gold = set().union(*(set(group) for group in alternatives)) if alternatives else set()
                    retrieved = {hit["id"] for hit in result["hits"]}
                    generated = set(result["generation_evidence_ids"])
                    row = {"id": question["id"], "category": question["category"], "method": method,
                        "document": question["document"], "question": question["question"], "gold_answer": question["gold_answer"],
                        "kind": question["kind"], "status": result["status"], "answer": result["answer"], "reason": result["reason"],
                        "raw_hit_any": int(bool(gold & retrieved)) if gold else "",
                        "raw_hit_all": int(complete(retrieved, alternatives)) if gold else "",
                        "generation_hit_all": int(complete(generated, alternatives)) if gold else "",
                        "answer_path": result["answer_path"], "latency_seconds": result["latency_seconds"],
                        "retrieval_query": result["retrieval_query"]["query"], "query_status": result["retrieval_query"]["status"],
                        "citation_ids": "|".join(c["id"] for c in result["citations"]),
                        "correct": "", "citation_supported": "", "failure_stage": "", "review_note": ""}
                    rows.append(row)
                    (folder / "review.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(question["id"], method, result["status"], "raw", row["raw_hit_all"], "sent", row["generation_hit_all"], result["answer"][:260].replace("\n", " "), flush=True)
        with (folder / "review.csv").open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("DIAGNOSTIC_RUN", folder, flush=True)
    finally:
        if engine:
            engine.worker.close()


if __name__ == "__main__":
    main()
