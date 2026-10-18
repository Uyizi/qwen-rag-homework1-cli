import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")


def main():
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer, Qwen3_5ForConditionalGeneration
    torch.set_num_threads(6)
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    start = time.perf_counter()
    etok = AutoTokenizer.from_pretrained(cfg["embedding_model"], local_files_only=True)
    emb = AutoModel.from_pretrained(cfg["embedding_model"], local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["generation_model"], local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg["generation_model"], local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda"}, attn_implementation="sdpa").eval()
    model.requires_grad_(False)
    metadata = {"ready": True, "model": cfg["generation_model"],
                "embedding_model": cfg["embedding_model"], "embedding_dimension": 384,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "python": sys.version.split()[0], "gpu": torch.cuda.get_device_name(),
                "load_seconds": round(time.perf_counter() - start, 3),
                "dtype": "bfloat16", "decoding": "greedy; thinking disabled"}
    print(json.dumps(metadata), flush=True)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            start = time.perf_counter()
            if req["op"] == "embed":
                rows = []
                texts = [req["prefix"] + x for x in req["texts"]]
                for offset in range(0, len(texts), 16):
                    inputs = etok(texts[offset:offset + 16], max_length=512,
                                  padding=True, truncation=True, return_tensors="pt")
                    with torch.inference_mode():
                        hidden = emb(**inputs).last_hidden_state
                        mask = inputs["attention_mask"].bool()
                        pooled = hidden.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1)[..., None]
                        rows.extend(torch.nn.functional.normalize(pooled, p=2, dim=1).tolist())
                result = {"vectors": rows}
            elif req["op"] == "generate":
                inputs = tokenizer.apply_chat_template(
                    req["messages"], tokenize=True, add_generation_prompt=True,
                    enable_thinking=False, return_tensors="pt", return_dict=True).to("cuda")
                limit = req.get("max_new_tokens", cfg["max_new_tokens"])
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=limit,
                                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
                ids = output[0, inputs["input_ids"].shape[1]:]
                result = {"text": tokenizer.decode(ids, skip_special_tokens=True).strip(),
                          "input_tokens": inputs["input_ids"].shape[1],
                          "output_tokens": len(ids), "output_at_limit": len(ids) == limit}
            else:
                raise ValueError("unknown operation")
            result["seconds"] = round(time.perf_counter() - start, 4)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
