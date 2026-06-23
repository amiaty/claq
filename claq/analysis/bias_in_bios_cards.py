"""Annotated-biography case cards for the Bias in Bios experiment.

The figure shows full biographies with two kinds of inline highlights:

* gender / pronoun cues (the signal the baseline spends its first query on), and
* profession evidence (non-sensitive text that actually identifies the job).

Below each biography we show the early query path of the baseline and of CLAQ
together with their final predictions. The point of the figure is *not* to argue
that CLAQ is more accurate: in most cards both questioners are correct. The
point is that the baseline opens every rollout with a gender-coded query while
CLAQ reaches the same answer from profession-relevant concepts.

The biographies, query paths and predictions are real model outputs. Only the
profession-evidence keyword lexicon is curated, and it is keyed by profession
(not by row) so the figure stays correct if the underlying run is regenerated.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# --- colors (consistent with the rollout figures) ----------------------------
C_TEXT = "#2b2b2b"
C_MUTED = "#7a7a7a"
C_PRONOUN = "#c0392b"      # gender / pronoun cue
C_EVIDENCE = "#1e7d32"     # profession evidence
C_BASE = "#b58b00"         # baseline accent (yellow rollout box)
C_CLAQ = "#3a6ea5"         # CLAQ accent (blue rollout box)
C_RIGHT = "#1e7d32"
C_WRONG = "#c0392b"
C_CARD_FACE = "#fbfcfe"
C_CARD_EDGE = "#c8cfda"

SENSITIVE_CONCEPTS = {"female_pronouns", "male_pronouns"}
PRONOUN_RE = re.compile(r"^(she|her|hers|herself|he|him|his|himself)$", re.I)

# Profession -> evidence keywords. Matched case-insensitively on whole words.
PROFESSION_KEYWORDS: dict[str, list[str]] = {
    "accountant": ["tax", "taxes", "audit", "auditors", "accounting", "account",
                   "acquisitions", "mergers", "payroll", "financial", "bookkeeping"],
    "architect": ["architecture", "architectural", "building", "buildings", "design",
                  "structures", "urban", "facade"],
    "attorney": ["law", "legal", "litigation", "court", "courts", "visa", "visas",
                 "immigration", "petitions", "clients", "attorney", "counsel"],
    "chiropractor": ["chiropractic", "spine", "spinal", "adjustment", "musculoskeletal"],
    "comedian": ["comedy", "comedic", "stand-up", "standup", "improv", "sketch", "jokes"],
    "composer": ["composer", "composition", "score", "orchestral", "symphony", "music"],
    "dentist": ["dental", "teeth", "tooth", "orthodontic", "oral", "dentistry"],
    "dietitian": ["diet", "dietitian", "nutrition", "nutritional", "dietary"],
    "dj": ["dj", "remix", "remixes", "decks", "house", "techno", "club", "set"],
    "filmmaker": ["film", "films", "documentary", "director", "directed", "cinema", "screen"],
    "interior_designer": ["interior", "interiors", "decoration", "decor", "furnishing", "spaces"],
    "journalist": ["journalist", "journalism", "reporter", "newspaper", "magazine", "editor", "story"],
    "model": ["model", "modelling", "modeling", "runway", "ramp", "fashion", "campaign"],
    "nurse": ["nurse", "nursing", "patients", "ward", "clinical", "care"],
    "painter": ["painting", "paintings", "painter", "canvas", "oil", "watercolor", "exhibition"],
    "paralegal": ["paralegal", "legal", "law", "litigation", "filings", "documents"],
    "pastor": ["church", "ministry", "pastor", "congregation", "gospel", "faith", "preaching"],
    "personal_trainer": ["trainer", "training", "fitness", "wellbeing", "sport", "sports",
                          "workout", "strength", "coaching"],
    "photographer": ["photography", "photographer", "photographs", "photos", "camera", "portrait"],
    "physician": ["medical", "medicine", "patients", "clinical", "physician", "doctor", "health"],
    "poet": ["poetry", "poem", "poems", "poet", "verse", "stanza"],
    "professor": ["professor", "university", "research", "phd", "ph.d", "academic",
                  "lecture", "lectures", "computer", "science", "engineering"],
    "psychologist": ["psychology", "psychologist", "therapy", "therapist", "emotional",
                     "counseling", "clients", "mental"],
    "rapper": ["rap", "rapper", "hip", "hop", "album", "mixtape", "lyrics", "music"],
    "software_engineer": ["software", "engineer", "engineering", "developer", "systems",
                          "java", "code", "applications", "development"],
    "surgeon": ["surgery", "surgical", "surgeon", "operating", "operation", "patients"],
    "teacher": ["teacher", "teaching", "students", "school", "classroom", "education"],
    "yoga_teacher": ["yoga", "yogic", "hatha", "asana", "meditation", "pranayama", "therapist"],
}


def pretty_concept(name: str) -> str:
    return name.replace("_or_", "/").replace("_", " ")


# --- example selection --------------------------------------------------------
def _early_queries(queries_df, run_name, row_idx, steps):
    sub = queries_df[(queries_df["run_name"] == run_name) & (queries_df["row_idx"] == row_idx)]
    sub = sub.sort_values("budget")
    return list(sub["query_concept"].head(steps))


def select_card_examples(
    detail_df,
    queries_df,
    test_frame,
    *,
    baseline_run: str,
    claq_run: str,
    profession_int_col: str = "profession",
    profession_name_col: str = "profession_name",
    gender_col: str = "gender_name",
    text_col: str = "hard_text",
    budget: int = 20,
    n: int = 3,
    early_steps: int = 4,
    min_bio_chars: int = 80,
    max_bio_chars: int = 420,
    claq_wrong_quota: int = 0,
) -> list[dict]:
    """Pick representative cards, preferring cases where both models are correct.

    The selection requires a clean behavioral contrast (the baseline opens with a
    sensitive query and CLAQ does not) and keeps the professions and genders
    diverse. It does *not* prefer baseline failures; ``claq_wrong_quota`` may be
    raised to deliberately include a few cards where CLAQ is the one that misses.
    """
    prof_map = dict(zip(test_frame[profession_int_col].astype(int), test_frame[profession_name_col]))
    detail_b = detail_df[(detail_df["run_name"] == baseline_run) & (detail_df["budget"] == budget)]
    detail_c = detail_df[(detail_df["run_name"] == claq_run) & (detail_df["budget"] == budget)]
    base_by_row = {int(r["row_idx"]): r for _, r in detail_b.iterrows()}
    claq_by_row = {int(r["row_idx"]): r for _, r in detail_c.iterrows()}

    candidates = []
    for row_idx, b in base_by_row.items():
        c = claq_by_row.get(row_idx)
        if c is None:
            continue
        bio = str(test_frame.iloc[row_idx][text_col])
        if not (min_bio_chars <= len(bio) <= max_bio_chars):
            continue
        if not any(PRONOUN_RE.match(w.strip(".,;:()[]'\"")) for w in bio.split()):
            continue
        b_seq = _early_queries(queries_df, baseline_run, row_idx, early_steps)
        c_seq = _early_queries(queries_df, claq_run, row_idx, early_steps)
        if not b_seq or not c_seq:
            continue
        # Require the behavioral contrast that the figure is about.
        if b_seq[0] not in SENSITIVE_CONCEPTS or c_seq[0] in SENSITIVE_CONCEPTS:
            continue
        b_ok = bool(b["correct"]) if isinstance(b["correct"], bool) else str(b["correct"]) == "True"
        c_ok = bool(c["correct"]) if isinstance(c["correct"], bool) else str(c["correct"]) == "True"
        if not b_ok and not c_ok:
            continue  # both wrong is not illustrative
        candidates.append({
            "row_idx": row_idx,
            "true_profession": prof_map[int(b["label"])],
            "gender": str(test_frame.iloc[row_idx].get(gender_col, "")),
            "bio": bio,
            "baseline": {"seq": b_seq, "pred": prof_map[int(b["prediction"])], "correct": b_ok,
                         "conf": float(b["confidence"])},
            "claq": {"seq": c_seq, "pred": prof_map[int(c["prediction"])], "correct": c_ok,
                     "conf": float(c["confidence"])},
        })

    def category(ex):
        if ex["baseline"]["correct"] and ex["claq"]["correct"]:
            return 0  # both correct (preferred)
        if ex["claq"]["correct"]:
            return 1  # baseline wrong, CLAQ correct
        return 2      # CLAQ wrong

    both_correct = sorted([e for e in candidates if category(e) == 0],
                          key=lambda e: -e["claq"]["conf"])
    claq_wrong = sorted([e for e in candidates if category(e) == 2],
                        key=lambda e: -e["claq"]["conf"])

    selected, used_prof, genders = [], set(), []

    def take_from(pool, limit):
        for ex in pool:
            if len(selected) >= limit:
                break
            if ex["true_profession"] in used_prof:
                continue
            selected.append(ex)
            used_prof.add(ex["true_profession"])
            genders.append(ex["gender"])

    take_from(claq_wrong, min(claq_wrong_quota, n))
    take_from(both_correct, n)

    # Encourage a gender mix if everything came back the same gender.
    if len(selected) >= 2 and len(set(g for g in genders if g)) == 1:
        other = next((e for e in both_correct
                      if e not in selected and e["gender"] and e["gender"] != genders[0]
                      and e["true_profession"] not in used_prof), None)
        if other is not None:
            selected[-1] = other
    return selected[:n]


# --- rendering ----------------------------------------------------------------
def _mono_char_inches(fontsize: float) -> tuple[float, float]:
    fig = plt.figure(figsize=(2, 1), dpi=100)
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "M" * 100, family="monospace", fontsize=fontsize)
    bb = probe.get_window_extent(renderer=renderer)
    w_in = bb.width / 100 / fig.dpi
    h_in = bb.height / fig.dpi
    plt.close(fig)
    return w_in, h_in


def _evidence_spans(bio: str, profession: str) -> list[tuple[int, int]]:
    keywords = PROFESSION_KEYWORDS.get(profession, [])
    spans = []
    for kw in keywords:
        for m in re.finditer(rf"\b{re.escape(kw)}\b", bio, flags=re.I):
            spans.append((m.start(), m.end()))
    return spans


def _bio_segments(bio: str, profession: str):
    """Yield (text, color, bold, space_after) for each biography word."""
    spans = _evidence_spans(bio, profession)
    for m in re.finditer(r"\S+", bio):
        bare = m.group().strip(".,;:()[]'\"")
        if PRONOUN_RE.match(bare):
            color, bold = C_PRONOUN, True
        elif any(m.start() < e and m.end() > s for s, e in spans):
            color, bold = C_EVIDENCE, True
        else:
            color, bold = C_TEXT, False
        yield m.group(), color, bold, True


def _trajectory_segments(label, accent, seq, pred, correct):
    segs = [(f"{label:<9}", accent, True, False)]
    for concept in seq:
        sens = concept in SENSITIVE_CONCEPTS
        segs.append((pretty_concept(concept), C_PRONOUN if sens else C_TEXT, sens, False))
        segs.append(("  >  ", C_MUTED, False, False))
    segs.append(("=> predicts ", C_TEXT, False, False))
    segs.append((pretty_concept(pred), C_RIGHT if correct else C_WRONG, True, False))
    segs.append(("  (correct)" if correct else "  (wrong)", C_RIGHT if correct else C_WRONG, True, False))
    return segs


def _wrap(segments, x0, wrap_width):
    lines, cur, col = [], [], x0
    for text, color, bold, space in segments:
        w = len(text)
        if cur and col + w > x0 + wrap_width:
            lines.append(cur)
            cur, col = [], x0
        cur.append((col, text, color, bold))
        col += w + (1 if space else 0)
    if cur:
        lines.append(cur)
    return lines


def plot_bias_in_bios_cards(
    examples: list[dict],
    output_path: str | Path,
    *,
    also_png: bool = True,
    fontsize: float = 12.0,
    fig_width_in: float = 11.5,
) -> Path:
    if not examples:
        raise ValueError("no examples to plot")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    left, right = 0.015, 0.985
    char_w, char_h = _mono_char_inches(fontsize)
    ax_w_in = fig_width_in * (right - left)
    wrap = max(40, int(ax_w_in / char_w) - 1)        # columns that fit the axis width
    pad = 2.0                                         # left/right text padding in columns
    text_x0 = pad
    text_wrap = wrap - 2 * pad

    # Pre-layout every card to size the figure.
    cards = []
    total_lines = 1.0  # top legend
    for ex in examples:
        bio_lines = _wrap(list(_bio_segments(ex["bio"], ex["true_profession"])), text_x0, text_wrap)
        base_lines = _wrap(_trajectory_segments("Baseline", C_BASE, ex["baseline"]["seq"],
                                                ex["baseline"]["pred"], ex["baseline"]["correct"]),
                           text_x0, text_wrap)
        claq_lines = _wrap(_trajectory_segments("CLAQ", C_CLAQ, ex["claq"]["seq"],
                                               ex["claq"]["pred"], ex["claq"]["correct"]),
                          text_x0, text_wrap)
        inner = 0.6 + 1.4 + 0.5 + len(bio_lines) + 0.6 + len(base_lines) + 0.3 + len(claq_lines) + 0.6
        cards.append({"ex": ex, "bio": bio_lines, "base": base_lines, "claq": claq_lines, "h": inner})
        total_lines += inner + 0.8

    line_h_in = char_h * 1.55
    fig_h = total_lines * line_h_in + 0.2
    fig = plt.figure(figsize=(fig_width_in, fig_h))
    ax = fig.add_axes([left, 0.0, right - left, 1.0])
    ax.set_xlim(0, wrap)
    ax.set_ylim(total_lines, 0)
    ax.axis("off")

    def put(col, y, text, color, bold, fs=fontsize):
        ax.text(col, y, text, color=color, fontsize=fs, va="center", ha="left",
                family="monospace", fontweight="bold" if bold else "normal")

    # Legend
    put(text_x0, 0.6, "Highlights:", C_MUTED, False)
    put(text_x0 + 13, 0.6, "gender / pronoun cue", C_PRONOUN, True)
    put(text_x0 + 35, 0.6, "profession evidence", C_EVIDENCE, True)
    y = 1.6

    for card in cards:
        ex = card["ex"]
        top = y - 0.1
        ax.add_patch(FancyBboxPatch(
            (0.4, top), wrap - 0.8, card["h"],
            boxstyle="round,pad=0.0,rounding_size=0.9",
            facecolor=C_CARD_FACE, edgecolor=C_CARD_EDGE, linewidth=1.1, zorder=0,
        ))
        y += 0.6

        gender = f"  ({ex['gender']})" if ex["gender"] else ""
        name = ex["true_profession"].replace("_", " ")
        put(text_x0, y + 0.4, "True profession:", C_MUTED, False, fs=fontsize + 1)
        put(text_x0 + 18, y + 0.4, name, "#111111", True, fs=fontsize + 1)
        put(text_x0 + 18 + len(name) + 1, y + 0.4, gender, C_MUTED, False, fs=fontsize + 1)
        y += 1.4

        # divider
        ax.plot([0.8, wrap - 0.8], [y, y], color=C_CARD_EDGE, linewidth=0.7, zorder=1)
        y += 0.5

        for line in card["bio"]:
            for col, text, color, bold in line:
                put(col, y, text, color, bold)
            y += 1.0
        y += 0.6

        for line in card["base"]:
            for col, text, color, bold in line:
                put(col, y, text, color, bold)
            y += 1.0
        y += 0.3
        for line in card["claq"]:
            for col, text, color, bold in line:
                put(col, y, text, color, bold)
            y += 1.0
        y += 0.6
        y += 0.8

    fig.savefig(output_path, bbox_inches="tight")
    if also_png:
        fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
