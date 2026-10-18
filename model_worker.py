import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

from chat_session import trim_chat_messages

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")


def chat_stop_token_ids(tokenizer, generation_config):
    configured = generation_config.eos_token_id
    values = configured if isinstance(configured, list) else [configured]
    return list(dict.fromkeys(value for value in [tokenizer.eos_token_id, *values] if value is not None))


def main():
    import torch
    import transformers
    from transformers import (AutoModel, AutoTokenizer, Qwen3_5ForConditionalGeneration,
                              TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList)
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
    requests = queue.Queue()
    cancellations = {}

    def read_requests():
        try:
            for line in sys.stdin:
                try:
                    req = json.loads(line)
                    key = req.get("request_id")
                    if req.get("op") == "cancel":
                        event = cancellations.get(key)
                        if event is not None:
                            event.set()
                    else:
                        if req.get("stream"):
                            cancellations[key] = threading.Event()
                        requests.put(req)
                except ValueError as exc:
                    requests.put({"op": "invalid", "detail": str(exc)})
        finally:
            for event in list(cancellations.values()):
                event.set()
            requests.put(None)

    threading.Thread(target=read_requests, daemon=True).start()
    while (req := requests.get()) is not None:
        key = req.get("request_id")
        try:
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
                limit = req.get("max_new_tokens", cfg["max_new_tokens"])
                messages = req["messages"]
                dropped = 0

                def encode(rows):
                    return tokenizer.apply_chat_template(
                        rows, tokenize=True, add_generation_prompt=True,
                        enable_thinking=False, return_tensors="pt", return_dict=True)

                if req.get("chat"):
                    context = min(req.get("context_tokens", 8192),
                                  getattr(model.config.text_config, "max_position_embeddings", 8192))
                    messages, dropped = trim_chat_messages(
                        messages, lambda rows: encode(rows)["input_ids"].shape[1], context - limit)
                inputs = encode(messages).to("cuda")
                if req.get("stream"):
                    cancel = cancellations[key]

                    class StopOnCancel(StoppingCriteria):
                        def __call__(self, input_ids, scores, **kwargs):
                            return cancel.is_set()

                    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True,
                                                    timeout=0.2)
                    outcome = {}

                    def generate():
                        try:
                            options = {"eos_token_id": chat_stop_token_ids(tokenizer, model.generation_config)} if req.get("chat") else {}
                            with torch.inference_mode():
                                outcome["output"] = model.generate(
                                    **inputs, max_new_tokens=limit, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id, streamer=streamer,
                                    stopping_criteria=StoppingCriteriaList([StopOnCancel()]), **options)
                        except Exception as exc:
                            outcome["error"] = exc
                            streamer.end()

                    thread = threading.Thread(target=generate, daemon=True)
                    thread.start()
                    while True:
                        try:
                            piece = next(streamer)
                            if piece and not cancel.is_set():
                                print(json.dumps({"event": "text", "request_id": key, "text": piece},
                                                 ensure_ascii=False), flush=True)
                        except queue.Empty:
                            continue
                        except StopIteration:
                            break
                    thread.join()
                    if "error" in outcome:
                        raise outcome["error"]
                    output = outcome["output"]
                else:
                    with torch.inference_mode():
                        output = model.generate(**inputs, max_new_tokens=limit,
                                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
                ids = output[0, inputs["input_ids"].shape[1]:]
                result = {"text": tokenizer.decode(ids, skip_special_tokens=True).strip(),
                          "input_tokens": inputs["input_ids"].shape[1],
                          "output_tokens": len(ids), "output_at_limit": len(ids) == limit}
                if req.get("stream"):
                    result.update(event="done", request_id=key, cancelled=cancel.is_set(), trimmed_turns=dropped)
            else:
                raise ValueError("unknown operation")
            result["seconds"] = round(time.perf_counter() - start, 4)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"error": str(exc), "request_id": key}, ensure_ascii=False), flush=True)
        finally:
            cancellations.pop(key, None)


if __name__ == "__main__":
    main()
