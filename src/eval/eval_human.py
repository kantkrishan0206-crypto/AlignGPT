#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Human evaluation utilities:
- Export a set of prompts and candidate outputs to JSONL/CSV
- Collect human ratings/preferences via:
    1) CLI mode (simple terminal prompts)
    2) Lightweight local web UI (Flask)
- Save human judgments to data/human_eval.jsonl
- Provide export in pref_pairs.jsonl format for DPO/RM retraining

Usage (export set):
  python src/eval/eval_human.py export \
    --prompts data/prompts.jsonl \
    --model-ckpt ./checkpoints/ppo_gpt2 \
    --model-name gpt2 \
    --output data/human_eval_candidates.jsonl \
    --num-per-prompt 2

Usage (CLI collect):
  python src/eval/eval_human.py cli \
    --candidates data/human_eval_candidates.jsonl \
    --output data/human_eval.jsonl

Usage (Web UI):
  python src/eval/eval_human.py web \
    --candidates data/human_eval_candidates.jsonl \
    --output data/human_eval.jsonl \
    --port 5000
"""

import os
import json
import argparse
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from src.utils.logging import init_logging, get_logger
from src.utils.seed import set_global_seed


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def export_candidates_for_human_eval(
    prompts_path: str,
    model_ckpt: str,
    model_name: str,
    output_path: str,
    num_per_prompt: int = 2,
    max_prompt_length: int = 256,
    max_new_tokens: int = 128,
    top_p: float = 0.9,
    temperature: float = 1.0,
):
    """
    Generate N candidates per prompt to be rated by humans.
    Schema: {"prompt": "...", "candidates": ["...", "..."], "n": 2}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_ckpt).to(device)
    model.eval()

    gen_cfg = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=num_per_prompt,
        return_dict_in_generate=True,
    )

    prompts = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            p = obj.get("prompt") or obj.get("instruction") or obj.get("text")
            if p:
                prompts.append(p)

    rows = []
    with torch.no_grad():
        for p in prompts:
            enc = tokenizer(p, truncation=True, max_length=max_prompt_length, padding=True, return_tensors="pt").to(device)
            out = model.generate(**enc, generation_config=gen_cfg)
            seqs = out.sequences
            # For num_return_sequences, sequences may be stacked; decode all
            texts = [tokenizer.decode(seqs[i], skip_special_tokens=True) for i in range(seqs.size(0))]
            # dedupe
            uniq = []
            seen = set()
            for t in texts:
                t = t.strip()
                if t and t not in seen:
                    seen.add(t)
                    uniq.append(t)
            rows.append({"prompt": p, "candidates": uniq, "n": len(uniq)})

    write_jsonl(output_path, rows)


def cli_collect(candidates_path: str, output_path: str):
    """
    Simple CLI loop to record preferences:
    Writes lines: {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    rows = read_jsonl(candidates_path)
    out = []
    print("Starting CLI human evaluation. Press 1 or 2 to select preferred candidate; q to quit.")
    for row in rows:
        prompt = row.get("prompt")
        cands = row.get("candidates", [])
        if len(cands) < 2:
            continue
        # Present the first two candidates
        a, b = cands[0], cands[1]
        print("\nPrompt:\n", prompt)
        print("\nCandidate 1:\n", a)
        print("\nCandidate 2:\n", b)
        sel = input("\nChoose preferred (1/2, or q to quit): ").strip().lower()
        if sel == "q":
            break
        if sel == "1":
            out.append({"prompt": prompt, "chosen": a, "rejected": b})
        elif sel == "2":
            out.append({"prompt": prompt, "chosen": b, "rejected": a})
        else:
            print("Invalid input; skipping.")
    write_jsonl(output_path, out)
    print(f"Wrote {len(out)} human preferences to {output_path}")


def web_collect(candidates_path: str, output_path: str, port: int = 5000):
    """
    Lightweight Flask app to collect preferences.
    GET / -> list prompts one by one
    POST /vote -> record preferred candidate
    """
    from flask import Flask, request, render_template_string, redirect

    rows = read_jsonl(candidates_path)
    app = Flask(__name__)
    votes: List[Dict[str, Any]] = []

    template = """
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Human Evaluation</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .prompt { padding: 10px; background: #f0f0f0; }
            .candidate { margin-top: 15px; padding: 10px; border: 1px solid #ddd; }
            .buttons { margin-top: 20px; }
            button { padding: 8px 12px; margin-right: 10px; }
        </style>
    </head>
    <body>
        <h2>Human Evaluation</h2>
        {% if row %}
            <div class="prompt"><b>Prompt:</b><br/>{{ row.prompt }}</div>
            <div class="candidate"><b>Candidate A:</b><br/>{{ row.candidates[0] }}</div>
            <div class="candidate"><b>Candidate B:</b><br/>{{ row.candidates[1] }}</div>
            <div class="buttons">
                <form method="post" action="/vote">
                    <input type="hidden" name="index" value="{{ idx }}"/>
                    <button type="submit" name="choice" value="A">Choose A</button>
                    <button type="submit" name="choice" value="B">Choose B</button>
                </form>
            </div>
            <p>Progress: {{ idx+1 }}/{{ total }}</p>
        {% else %}
            <p>All done. Thank you!</p>
            <p><a href="/export">Download results</a></p>
        {% endif %}
    </body>
    </html>
    """

    @app.route("/", methods=["GET"])
    def index():
        idx = int(request.args.get("i", "0"))
        if idx >= len(rows):
            return render_template_string(template, row=None, idx=idx, total=len(rows))
        row = rows[idx]
        if len(row.get("candidates", [])) < 2:
            return redirect(f"/?i={idx+1}")
        return render_template_string(template, row=row, idx=idx, total=len(rows))

    @app.route("/vote", methods=["POST"])
    def vote():
        idx = int(request.form.get("index"))
        choice = request.form.get("choice")
        row = rows[idx]
        a, b = row["candidates"][0], row["candidates"][1]
        if choice == "A":
            votes.append({"prompt": row["prompt"], "chosen": a, "rejected": b})
        elif choice == "B":
            votes.append({"prompt": row["prompt"], "chosen": b, "rejected": a})
        return redirect(f"/?i={idx+1}")

    @app.route("/export", methods=["GET"])
    def export():
        write_jsonl(output_path, votes)
        return f"Saved {len(votes)} votes to {output_path}"

    app.run(host="127.0.0.1", port=port, debug=False)


def export_pref_pairs_from_human_eval(human_eval_path: str, output_path: str):
    """
    Convert collected human eval to pref_pairs.jsonl schema (already matches).
    """
    rows = read_jsonl(human_eval_path)
    # Already in desired schema: {"prompt": "...", "chosen": "...", "rejected": "..."}
    write_jsonl(output_path, rows)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    p_export = sub.add_parser("export")
    p_export.add_argument("--prompts", required=True)
    p_export.add_argument("--model-ckpt", required=True)
    p_export.add_argument("--model-name", default="gpt2")
    p_export.add_argument("--output", required=True)
    p_export.add_argument("--num-per-prompt", type=int, default=2)
    p_export.add_argument("--max-prompt-length", type=int, default=256)
    p_export.add_argument("--max-new-tokens", type=int, default=128)
    p_export.add_argument("--top-p", type=float, default=0.9)
    p_export.add_argument("--temperature", type=float, default=1.0)
    p_export.add_argument("--seed", type=int, default=42)

    p_cli = sub.add_parser("cli")
    p_cli.add_argument("--candidates", required=True)
    p_cli.add_argument("--output", required=True)

    p_web = sub.add_parser("web")
    p_web.add_argument("--candidates", required=True)
    p_web.add_argument("--output", required=True)
    p_web.add_argument("--port", type=int, default=5000)

    p_pref = sub.add_parser("export_pairs")
    p_pref.add_argument("--human-eval", required=True)
    p_pref.add_argument("--output", required=True)

    args = ap.parse_args()
    init_logging("./logs", run_name="eval_human")

    if args.cmd == "export":
        set_global_seed(args.seed)
        export_candidates_for_human_eval(
            prompts_path=args.prompts,
            model_ckpt=args.model_ckpt,
            model_name=args.model_name,
            output_path=args.output,
            num_per_prompt=args.num_per_prompt,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        print(f"Exported candidates for human eval to {args.output}")

    elif args.cmd == "cli":
        cli_collect(args.candidates, args.output)

    elif args.cmd == "web":
        web_collect(args.candidates, args.output, port=args.port)

    elif args.cmd == "export_pairs":
        export_pref_pairs_from_human_eval(args.human_eval, args.output)
        print(f"Exported pref pairs to {args.output}")

    else:
        raise ValueError("Use one of subcommands: export | cli | web | export_pairs")


if __name__ == "__main__":
    main()