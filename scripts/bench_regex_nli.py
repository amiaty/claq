"""Run only the regex + NLI benchmark logic from the notebook, locally.

Intended as a quick sanity check on Mac (MPS/CPU). Throughput numbers from
this run are NOT representative of the CUDA server — they only confirm the
code path is correct.
"""

from __future__ import annotations

import gc
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
CONCEPTS_CSV = REPO_ROOT / "assets" / "concepts" / "bias_in_bios.csv"
TEXT_COLUMN = "hard_text"
PROBE_SIZE = 128


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


LEXICAL_PATTERNS = {
    "direct_gender_pronouns": re.compile(
        r"\b(he|him|his|she|her|hers|himself|herself)\b", re.IGNORECASE
    ),
    "gendered_titles": re.compile(r"\b(Mr|Mrs|Ms|Miss|Sir|Madam|Madame)\.?\b"),
}


def regex_label_texts(texts, concept_names, lexical_idx):
    matrix = np.zeros((len(texts), len(concept_names)), dtype=np.float32)
    for concept_name, pattern in LEXICAL_PATTERNS.items():
        col = lexical_idx[concept_name]
        matrix[:, col] = [1.0 if pattern.search(t) else 0.0 for t in texts]
    return matrix


def run_regex_benchmark(texts, concept_names, lexical_idx):
    start = time.perf_counter()
    scores = regex_label_texts(texts, concept_names, lexical_idx)
    elapsed = max(time.perf_counter() - start, 1e-9)
    print(
        f"[regex] {len(texts)} bios in {elapsed * 1000:.1f} ms"
        f"  ({len(texts) / elapsed:,.0f} bios/sec)."
    )
    for name in LEXICAL_PATTERNS:
        col = lexical_idx[name]
        rate = float(scores[:, col].mean())
        positive_rows = [i for i in range(len(texts)) if scores[i, col]]
        sample = positive_rows[0] if positive_rows else None
        snippet = texts[sample][:160] if sample is not None else "(no matches)"
        print(f"  {name}: yes_rate={rate:.3f}  example: {snippet}")
    return scores


def run_nli_benchmark(
    texts,
    hypotheses,
    concept_names,
    device,
    model_name="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    batch_size=32,
    max_length=256,
):
    print(f"[nli] loading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(device).eval()
    entail_id = model.config.label2id.get(
        "entailment", model.config.label2id.get("ENTAILMENT", 0)
    )
    print(f"[nli] label2id={model.config.label2id}; using entail_id={entail_id}")

    n_text, n_hyp = len(texts), len(hypotheses)
    premises = [t for t in texts for _ in range(n_hyp)]
    hyps = list(hypotheses) * n_text
    flat = torch.empty(n_text * n_hyp, dtype=torch.float32)

    start = time.perf_counter()
    with torch.inference_mode():
        for s in range(0, len(premises), batch_size):
            e = s + batch_size
            enc = tokenizer(
                premises[s:e],
                hyps[s:e],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.float()
            flat[s:e] = torch.softmax(logits, dim=-1)[:, entail_id].cpu()
    elapsed = time.perf_counter() - start

    scores = flat.view(n_text, n_hyp)
    pairs = n_text * n_hyp
    pairs_per_sec = pairs / elapsed
    full_pairs = 257_478 * len(concept_names)
    full_hours = full_pairs / pairs_per_sec / 3600
    print(
        f"[nli] probe: {n_text} bios x {n_hyp} concepts = {pairs:,} pairs"
        f"  in {elapsed:.1f}s  ({pairs_per_sec:,.1f} pairs/sec)."
    )
    print(
        f"[nli] full-run estimate (this hardware): {full_pairs:,} pairs"
        f"  -> {full_hours:.2f}h"
    )
    return scores, pairs_per_sec, full_hours


def main():
    device = pick_device()
    print(f"device={device}, torch={torch.__version__}")

    concepts_df = pd.read_csv(CONCEPTS_CSV)
    concept_names = concepts_df["concept"].tolist()
    concept_texts = concepts_df["description"].tolist()
    lexical_idx = {n: concept_names.index(n) for n in LEXICAL_PATTERNS}
    print(f"loaded {len(concept_names)} concepts from {CONCEPTS_CSV}")

    print("loading LabHC/bias_in_bios train split (cached)...")
    raw_train = load_dataset("LabHC/bias_in_bios", split="train")
    probe_texts = [raw_train[i][TEXT_COLUMN] for i in range(PROBE_SIZE)]
    print(f"probe size={len(probe_texts)} bios")

    print("\n=== Regex benchmark ===")
    regex_scores = run_regex_benchmark(probe_texts, concept_names, lexical_idx)

    print("\n=== NLI benchmark ===")
    nli_hypotheses = [d.strip().rstrip(".").capitalize() + "." for d in concept_texts]
    nli_scores, pairs_per_sec, full_hours = run_nli_benchmark(
        probe_texts, nli_hypotheses, concept_names, device
    )

    print("\n=== Per-concept yes-rate (NLI @ 0.5 vs regex) ===")
    nli_yes_rate = (nli_scores >= 0.5).float().mean(dim=0).numpy()
    nli_mean_prob = nli_scores.mean(dim=0).numpy()
    df = pd.DataFrame(
        {
            "concept": concept_names,
            "kind": concepts_df["kind"].tolist(),
            "nli_yes_rate@0.5": nli_yes_rate,
            "nli_mean_prob": nli_mean_prob,
        }
    )
    df["regex_yes_rate"] = float("nan")
    for name, col in lexical_idx.items():
        df.loc[df["concept"] == name, "regex_yes_rate"] = float(
            regex_scores[:, col].mean()
        )
    df_sorted = df.sort_values("nli_yes_rate@0.5", ascending=False)
    print(df_sorted.to_string(index=False))

    flagged = df.query("`nli_yes_rate@0.5` > 0.75 and kind == 'utility'")
    if not flagged.empty:
        print("\nSuspicious utility concepts (NLI yes_rate > 0.75):")
        print(flagged.to_string(index=False))

    del nli_scores
    gc.collect()


if __name__ == "__main__":
    main()
