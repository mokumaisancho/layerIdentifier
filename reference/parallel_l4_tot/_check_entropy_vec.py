import json

with open("results/bench_max_tokens_512/benchmark_parallel.jsonl") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        if r["question_idx"] == 103:
            for i, p in enumerate(r["parallel"]["paths"]):
                ev = p.get("entropy_vec")
                feat = p.get("features", {})
                print(f"Q103 P{i}: entropy_vec type={type(ev).__name__}, val={ev}")
                print(f"  features keys: {list(feat.keys())}")
            break
