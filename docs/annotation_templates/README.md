# Gold Annotation Templates

Fill-in templates for the human gold-standard files defined in
[`../Caldera Engine Human Annotation Guidelines.md`](../Caldera%20Engine%20Human%20Annotation%20Guidelines.md).
Copy a template, replace the illustrative example entries with real annotations, and fill the
`_meta` block. Keys prefixed with `_` (`_meta`, `_instructions`, `_style_vocabulary`) are
guidance, not data — the eval harnesses ignore them.

| Template | Dimension | Output file | Scored by |
|---|---|---|---|
| `segmentation_gold.template.json` | 0 · Segmentation | `segmentation_gold.json` | (foundation — aligns all others) |
| `structure_gold.template.json` | 1 · Structure | `structure_gold.json` | boundary P/R/F1, scene-count, text fidelity |
| `attribution_gold.template.json` | 2 · Attribution | `attribution_gold.json` | speaker + alias accuracy |
| `reduction_gold.template.json` | 3 · Attribution reduction | `reduction_gold.json` | remove/keep precision, delivery extraction |
| `emotion_gold.template.json` | 4 · Emotion | `emotion_gold.json` | per-line accuracy, distribution similarity |
| `expression_gold.template.json` | 5 · Expression | `expression_gold.json` | **`python -m src.eval_expression`** |
| `production_gold.template.json` | 6 · Production | `production_gold.json` | human rating, cue/placement accuracy |

## Running the expression eval (Dimension 5)

Once `expression_gold.json` exists and the character designer has produced the book's
`character_profiles.json`:

```bash
python -m src.eval_expression "A Scandal in Bohemia" --gold path/to/expression_gold.json
```

Reports exact-style agreement, band agreement (reserved/balanced/intensified near-miss),
coverage gaps, evidence-support rate, and confidence calibration. Add `--json` for the raw
result. Both sides are canonicalized through the engine's own vocabulary, so noun emotions
("anger") and style synonyms ("stoic") never cause a false mismatch.

The per-line emotion eval (Dimension 4) and the qualitative distribution report are covered by
`tests/test_emotion_distribution.py` and `python -m src.eval_emotion "<book>"`.
