import json
import argparse
from rouge_score import rouge_scorer
import bert_score
import numpy as np

# === Argument parser ===
parser = argparse.ArgumentParser(description="Evaluate summaries using ROUGE and BERTScore.")
parser.add_argument('--file', type=str, required=True, help='Path to the input JSON file')
parser.add_argument('--key', type=str, required=True, help='Key name for generated summaries')
parser.add_argument('--ref', type=str, required=True, help='Key name for generated summaries')
args = parser.parse_args()

# === Load JSON file ===
with open(args.file, "r", encoding="utf-8") as f:
    data = json.load(f)

# === Extract summaries and references ===
generated = [item[args.key] for item in data]
references = [item[args.ref] for item in data]

# === ROUGE evaluation ===
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_results = {'rouge1': [], 'rouge2': [], 'rougeL': []}

for ref, gen in zip(references, generated):
    scores = scorer.score(ref, gen)
    for key in rouge_results:
        rouge_results[key].append({
            'precision': scores[key].precision,
            'recall': scores[key].recall,
            'fmeasure': scores[key].fmeasure
        })

# === Compute average ROUGE ===
def avg_metric(metric_list, score_type):
    return np.mean([s[score_type] for s in metric_list])

avg_rouge_scores = {}
for key in rouge_results:
    avg_rouge_scores[key] = {
        'precision': avg_metric(rouge_results[key], 'precision'),
        'recall': avg_metric(rouge_results[key], 'recall'),
        'fmeasure': avg_metric(rouge_results[key], 'fmeasure')
    }

# === BERTScore ===
P, R, F1 = bert_score.score(generated, references, lang="zh", verbose=False)
bert_results = {
    'precision': P.mean().item(),
    'recall': R.mean().item(),
    'f1': F1.mean().item()
}

# === Print results ===
print("\n===== ROUGE =====")
for k, v in avg_rouge_scores.items():
    print(f"{k.upper()}: Precision={v['precision']:.4f}, Recall={v['recall']:.4f}, F1={v['fmeasure']:.4f}")

print("\n===== BERTScore =====")
print(f"Precision={bert_results['precision']:.4f}, Recall={bert_results['recall']:.4f}, F1={bert_results['f1']:.4f}")
