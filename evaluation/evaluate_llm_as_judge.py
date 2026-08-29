import os
import json
import argparse
import pandas as pd


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="deepseek / gpt5 / mistral / phi4 / qwen2.5"
)

parser.add_argument(
    "--day",
    type=str,
    required=True,
    help="3day / 5day / 7day"
)

parser.add_argument(
    "--llm",
    action="store_true",
    help="Use llm_eval files"
)

args = parser.parse_args()


# ============================================================
# FIND TSV FILE
# ============================================================

FOLDER = "/scratch/sg/Vijay/TripCraft/llm-as-judge_tsvs"

if args.llm:
    pattern = f"{args.model}_llm_eval_{args.day}"
else:
    pattern = f"{args.model}_eval_{args.day}"

matched_files = sorted([
    x for x in os.listdir(FOLDER)
    if pattern in x and x.endswith(".tsv")
])

if len(matched_files) == 0:

    print(f"\n[!] No TSV found for pattern: {pattern}")
    exit()

file = matched_files[0]

path = os.path.join(FOLDER, file)

print("\n" + "=" * 80)
print(file)
print("=" * 80)


# ============================================================
# LOAD TSV
# ============================================================

df = pd.read_csv(
    path,
    sep="\t"
)

print(f"\n[*] Loaded rows: {len(df)}")


# ============================================================
# STORAGE
# ============================================================

pairwise_counts = {
    "with_review_strong_win": 0,
    "with_review_win": 0,
    "without_review_strong_win": 0,
    "without_review_win": 0,
    "tie": 0
}

score_storage = {
    "with_review": {
        "persona_alignment": [],
        "experiential_quality": [],
        "risk_avoidance": [],
        "overall_satisfaction": []
    },

    "without_review": {
        "persona_alignment": [],
        "experiential_quality": [],
        "risk_avoidance": [],
        "overall_satisfaction": []
    }
}


# ============================================================
# PROCESS ROWS
# ============================================================

for idx, row in df.iterrows():

    response = row.get("evaluation_response", "")

    if pd.isna(response):
        continue

    response = str(response).strip()

    if response == "":
        continue

    try:

        data = json.loads(response)

    except Exception:

        print(f"[!] JSON parse failed at row {idx}")
        continue

    # ========================================================
    # LABELS
    # ========================================================

    a_label = row["a_label"]
    b_label = row["b_label"]

    # ========================================================
    # SCORES
    # ========================================================

    metrics = [
        "persona_alignment",
        "experiential_quality",
        "risk_avoidance",
        "overall_satisfaction"
    ]

    for metric in metrics:

        try:

            a_score = data[metric]["A"]
            b_score = data[metric]["B"]

            score_storage[a_label][metric].append(a_score)
            score_storage[b_label][metric].append(b_score)

        except:
            pass

    # ========================================================
    # PREFERENCE
    # ========================================================

    pref = str(
        data.get("pairwise_preference", "")
    ).strip().lower()

    if pref == "tie":

        pairwise_counts["tie"] += 1

    elif "strongly" in pref:

        if pref.startswith("a"):

            if a_label == "with_review":
                pairwise_counts["with_review_strong_win"] += 1
            else:
                pairwise_counts["without_review_strong_win"] += 1

        elif pref.startswith("b"):

            if b_label == "with_review":
                pairwise_counts["with_review_strong_win"] += 1
            else:
                pairwise_counts["without_review_strong_win"] += 1

    else:

        if pref.startswith("a"):

            if a_label == "with_review":
                pairwise_counts["with_review_win"] += 1
            else:
                pairwise_counts["without_review_win"] += 1

        elif pref.startswith("b"):

            if b_label == "with_review":
                pairwise_counts["with_review_win"] += 1
            else:
                pairwise_counts["without_review_win"] += 1


# ============================================================
# COMPUTE WIN RATE
# ============================================================

total = sum(pairwise_counts.values())

with_review_total_wins = (
    pairwise_counts["with_review_strong_win"] +
    pairwise_counts["with_review_win"]
)

win_rate = (
    100 * with_review_total_wins / total
    if total > 0 else 0
)


# ============================================================
# COMPUTE SCORES
# ============================================================

def avg(vals):

    if len(vals) == 0:
        return 0

    return round(sum(vals) / len(vals), 2)


persona_score = avg(
    score_storage["with_review"]["persona_alignment"]
)

experience_score = avg(
    score_storage["with_review"]["experiential_quality"]
)

satisfaction_score = avg(
    score_storage["with_review"]["overall_satisfaction"]
)


# ============================================================
# PRETTY MODEL NAME
# ============================================================

model_map = {
    "gpt5": "GPT-5",
    "phi4": "Phi-4",
    "qwen2.5": "Qwen 2.5",
    "llama3.1": "Llama 3.1 (70B)",
    "deepseek": "DeepSeek-R1 (14B)",
    "mistral": "Mistral-Nemo"
}

pretty_model = model_map.get(
    args.model,
    args.model
)


# ============================================================
# PRINT LATEX ROW
# ============================================================

print("\n" + "=" * 80)
print("LATEX TABLE ROW")
print("=" * 80)

print(
    f"{pretty_model} "
    f"& {args.day.replace('day', '-day')} "
    f"& {win_rate:.2f} "
    f"& {persona_score:.2f} "
    f"& {experience_score:.2f} "
    f"& {satisfaction_score:.2f} \\\\"
)

print("=" * 80)