#!/usr/bin/env python3
"""
LLM-as-judge over the A/B evaluation TSV, calling Qwen/Qwen3-32B through the
Hugging Face Inference interface (huggingface_hub.InferenceClient).

Reads the TSV from prompt_tsv.py, fills prompt.txt per row, calls Qwen3-32B,
parses the strict-JSON verdict into gpt_response, adds flattened score columns,
and de-blinds A/B (via a_label/b_label) into a with_review vs without_review
win rate.

Setup
-----
pip install "huggingface_hub>=0.25" pandas
export HF_TOKEN=hf_xxx            # your token (needs Inference Providers access)

Run
---
python judge_qwen_hf.py \
    --input_tsv gpt5_llm_eval_3day.tsv \
    --output_tsv gpt5_llm_eval_3day_judged.tsv \
    --prompt_file prompt.txt \
    --workers 4                  # keep modest to respect rate limits

Notes
-----
* Provider routing: --provider auto lets HF pick an available provider that
  serves Qwen3-32B (Together / Fireworks / Novita / Hyperbolic / ...). You can
  pin one, e.g. --provider together.
* Thinking: Qwen3 emits <think>...</think> by default. We append "/no_think"
  (Qwen3's soft switch) so the judge returns clean JSON cheaply. Pass
  --enable_thinking to keep the reasoning trace; the parser strips it either way.
* Resume: re-running with --resume skips rows already judged in --output_tsv.
"""
import os, re, json, time, argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PLACEHOLDERS = ["persona", "itinerary_a", "reviews_a", "itinerary_b", "reviews_b"]


# ----------------------------------------------------------------------
# Prompt building  (template has literal { } braces -> no str.format)
# ----------------------------------------------------------------------
def load_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(template, row, enable_thinking):
    out = template
    for key in PLACEHOLDERS:
        out = out.replace("{" + key + "}", str(row.get(key, "") or ""))
    # Qwen3 soft switch for thinking mode
    out += "\n\n" + ("/think" if enable_thinking else "/no_think")
    return out


# ----------------------------------------------------------------------
# Robust JSON extraction
# ----------------------------------------------------------------------
def extract_json(text):
    if not text:
        return None, "empty"
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    if start == -1:
        return None, "no_brace"
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1]), "ok"
                except Exception as e:
                    return None, f"parse_error: {e}"
    return None, "unbalanced"


# ----------------------------------------------------------------------
# HF Inference call (with retries)
# ----------------------------------------------------------------------
def make_client(args):
    from huggingface_hub import InferenceClient
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise SystemExit("No HF token found. Set --hf_token or export HF_TOKEN.")
    return InferenceClient(provider=args.provider, api_key=token, timeout=args.timeout)


def call_model(client, prompt, args):
    last_err = None
    for attempt in range(args.max_retries):
        try:
            resp = client.chat_completion(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            return resp.choices[0].message.content, None
        except Exception as e:
            last_err = str(e)
            # backoff on rate limit / transient errors
            time.sleep(min(2 ** attempt, 30))
    return None, last_err


# ----------------------------------------------------------------------
# De-blind A/B -> winning condition
# ----------------------------------------------------------------------
PREF_TO_SIDE = {
    "a strongly preferred": "A", "a preferred": "A",
    "b strongly preferred": "B", "b preferred": "B",
    "tie": "TIE",
}


def winner_condition(pref, a_label, b_label):
    if not isinstance(pref, str):
        return None
    side = PREF_TO_SIDE.get(pref.strip().lower())
    if side is None:
        return None
    if side == "TIE":
        return "tie"
    return a_label if side == "A" else b_label


FLAT_COLS = [
    "persona_alignment_A", "persona_alignment_B",
    "experiential_quality_A", "experiential_quality_B",
    "risk_avoidance_A", "risk_avoidance_B",
    "overall_satisfaction_A", "overall_satisfaction_B",
    "pairwise_preference", "reasoning", "parse_status", "winner_condition",
]


def flatten(obj, a_label, b_label):
    def g(k, s):
        try:
            return obj[k][s]
        except Exception:
            return None
    pref = obj.get("pairwise_preference")
    return {
        "persona_alignment_A": g("persona_alignment", "A"),
        "persona_alignment_B": g("persona_alignment", "B"),
        "experiential_quality_A": g("experiential_quality", "A"),
        "experiential_quality_B": g("experiential_quality", "B"),
        "risk_avoidance_A": g("risk_avoidance", "A"),
        "risk_avoidance_B": g("risk_avoidance", "B"),
        "overall_satisfaction_A": g("overall_satisfaction", "A"),
        "overall_satisfaction_B": g("overall_satisfaction", "B"),
        "pairwise_preference": pref,
        "reasoning": obj.get("reasoning"),
        "winner_condition": winner_condition(pref, a_label, b_label),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_tsv", required=True)
    ap.add_argument("--output_tsv", required=True)
    ap.add_argument("--prompt_file", required=True)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--provider", default="auto",
                    help="HF Inference provider: auto|together|fireworks-ai|novita|hyperbolic|...")
    ap.add_argument("--hf_token", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--resume", action="store_true",
                    help="Skip rows already judged in an existing --output_tsv.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    template = load_template(args.prompt_file)
    df = pd.read_csv(args.input_tsv, sep="\t")
    if args.limit:
        df = df.head(args.limit).copy()
    for c in FLAT_COLS:
        if c not in df.columns:
            df[c] = None
    if "gpt_response" not in df.columns:
        df["gpt_response"] = None
    # Ensure string/result columns are object dtype (they may load as all-NaN float64),
    # otherwise assigning a string raises a dtype-incompatibility error in newer pandas.
    _obj_cols = ["gpt_response", "parse_status", "pairwise_preference",
                 "reasoning", "winner_condition"]
    for c in _obj_cols:
        if c in df.columns:
            df[c] = df[c].astype(object)

    # resume: carry over previously judged rows
    done = set()
    if args.resume and os.path.exists(args.output_tsv):
        prev = pd.read_csv(args.output_tsv, sep="\t")
        prev_ok = prev[prev.get("parse_status").astype(str) == "ok"] if "parse_status" in prev else prev.iloc[0:0]
        prev_map = {str(r["index"]): r for _, r in prev_ok.iterrows()}
        for i in df.index:
            key = str(df.at[i, "index"])
            if key in prev_map:
                for c in ["gpt_response"] + FLAT_COLS:
                    if c in prev_map[key]:
                        df.at[i, c] = prev_map[key][c]
                done.add(i)
        print(f"[*] Resume: {len(done)} rows already judged, skipping them.")

    todo = [i for i in df.index if i not in done]
    print(f"[*] Judging {len(todo)} rows with {args.model} via HF ({args.provider}), "
          f"workers={args.workers}, thinking={'on' if args.enable_thinking else 'off'}")

    client = make_client(args)

    def work(i):
        prompt = build_prompt(template, df.loc[i], args.enable_thinking)
        txt, err = call_model(client, prompt, args)
        if txt is None:
            return i, f"__ERROR__ {err}", None, "call_failed"
        obj, status = extract_json(txt)
        return i, txt, obj, status

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, i): i for i in todo}
        for fut in as_completed(futures):
            i, txt, obj, status = fut.result()
            df.at[i, "gpt_response"] = txt if obj is None else json.dumps(obj, ensure_ascii=False)
            df.at[i, "parse_status"] = status
            if obj is not None:
                for k, v in flatten(obj, df.at[i, "a_label"], df.at[i, "b_label"]).items():
                    df.at[i, k] = v
            completed += 1
            if completed % args.save_every == 0:
                df.to_csv(args.output_tsv, sep="\t", index=False)
                print(f"    ...{completed}/{len(todo)} done (checkpoint saved)")

    df.to_csv(args.output_tsv, sep="\t", index=False)
    print(f"[✓] Saved -> {args.output_tsv}")

    # summary
    ok = (df["parse_status"].astype(str) == "ok").sum()
    print(f"[*] Parsed OK: {ok}/{len(df)}")
    vc = df["winner_condition"].dropna().value_counts()
    print("\n=== Pairwise outcome (de-blinded) ===")
    for cond, n in vc.items():
        print(f"  {cond:16s}: {n}")
    wr, wo = int(vc.get("with_review", 0)), int(vc.get("without_review", 0))
    if wr + wo:
        print(f"\n  with_review win rate (excl. ties): {wr}/{wr+wo} = {wr/(wr+wo)*100:.1f}%")


if __name__ == "__main__":
    main()