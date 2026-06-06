"""CLI dispatcher: emit-check | sweep.

Usage:
    python -m layerIdentifier.src.cli emit-check \\
        --model /path/to/your/model.mlx \\
        --out results/your-model/

    python -m layerIdentifier.src.cli sweep \\
        --model /path/to/your/model.mlx \\
        --out results/your-model/ \\
        --limit 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import emit_check, sweep


def main():
    ap = argparse.ArgumentParser(prog="layerIdentifier")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("emit-check", help="cheap entropy-emission probe (5 problems, all layers)")
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--max-tokens", type=int, default=256)
    e.add_argument("--temperature", type=float, default=0.1)

    s = sub.add_parser("sweep", help="full layer-ID sweep + ranking")
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max-tokens", type=int, default=150)
    s.add_argument("--temperature", type=float, default=0.1)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--chat-template", action="store_true",
                   help="apply tokenizer chat_template before generation")
    s.add_argument("--no-thinking", action="store_true",
                   help="disable thinking mode (set enable_thinking=False in chat template)")
    s.add_argument("--stop-sequences", type=str, default=None,
                   help="comma-separated decode-string stops (e.g. '<turn|>,<|end|>' for Gemma, '<|im_end|>' for Qwen)")

    args = ap.parse_args()
    if args.cmd == "emit-check":
        verdict = emit_check.run_emit_check(
            args.model, Path(args.out),
            max_tokens=args.max_tokens, temperature=args.temperature,
        )
        # Stdout summary
        print(json.dumps({k: v for k, v in verdict.items() if k != "captures"}, indent=2))
    elif args.cmd == "sweep":
        stop_sequences = None
        if args.stop_sequences:
            stop_sequences = [s.strip() for s in args.stop_sequences.split(",") if s.strip()]
        summary = sweep.run_sweep(
            args.model, Path(args.out),
            max_tokens=args.max_tokens, temperature=args.temperature,
            limit=args.limit,
            use_chat_template=args.chat_template,
            enable_thinking=not args.no_thinking,
            stop_sequences=stop_sequences,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
