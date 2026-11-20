# Harmonizer: a neurosymbolic melody harmonizer

Diploma thesis project, School of Electrical and Computer Engineering, National Technical University of Athens (AI and Learning Systems Lab). Supervisor: Prof. Athanasios Voulodimos.

An encoder-only Transformer (~25.5M parameters) reads a melody and predicts, for every note, a chord **root** (13 classes) and **quality** (24 classes). Music-theory rules are added to the training loss as differentiable penalties computed from the softmax probabilities. The repo contains the training code, the data extractors, the ablation/evaluation code, the results and the thesis chapters.

## Data

21,504 songs (2,559,122 notes) from four sources: PDMX, OpenEWLD, CMD and POP909. Split: 17,116 train / 4,280 val (classics 3,999, CMD 101, POP909 180 in validation). The data itself is not stored here (large). `src/extract_chord_melody_dataset.py` and `src/extract_pop909.py` produce the JSON files; `final_corpus.pkl` and `symbolic_lookup.pkl` are the pre-built corpus and vocabulary that `train_harmonizer.py` loads. The PDMX/OpenEWLD extraction step that builds those two pickles is not included in this repository.

## Rules

| Rule | Meaning |
|---|---|
| 1, 2, 5 | vocabulary / chord-tone consistency, leading-tone resolution |
| 3 | harmonic-rhythm stability |
| 6, 8 | functional harmony (T/S/D) and chord quality per function |
| 7 | piece ends on the tonic |
| 4 | disabled (conflicts with Rule 6) |

Penalty weights are λ = 1/(r+ε) − 1, computed from how often each rule is satisfied in the data.

## Main results (validation set, 200 epochs, last-epoch checkpoint)

Joint (root and quality both correct) accuracy per note:

| Variant | Joint acc. (%) |
|---|---|
| baseline (no rules) | 63.32 (second seed: 62.98) |
| full (all rules) | 65.31 (65.55) |
| full − {1,2,5} | 66.51 (66.20) |
| full − {3,7} | 65.80 |
| full − {6,8} | 65.87 |
| only {1,2,5} | 65.20 |
| only {3,7} | 66.12 |
| only {6,8} | 64.39 |

Averaged over two seeds, full vs. baseline is +2.29 pp (paired bootstrap 95% CI [+1.62, +2.96]; training-noise s ≈ 0.21 pp). Full CSVs are in `results/`, figures in `figures/`.

Known limitations (discussed in thesis chapters 9–10):
- Rule 7 overshoots: rule-trained models end on the tonic 99.6–99.9% of the time vs. 73.7% in the data.
- Rule 3 compares chord IRIs, so targets are miscalibrated for ~13.7% of notes (CMD/POP909).
- 8,559 POP909 notes were dropped (double sharps); CMD octaves are not shifted after transposition.
- Class weights are computed on the whole corpus before the split; duplicate songs across train/val (PDMX/MuseScore) were not checked.
- Only a subset of the variants has a second seed, so most pairwise differences between rule subsets are indicative only (Holm correction over 11 comparisons).

## Usage

```bash
pip install -r requirements.txt

# paths (defaults are the original Kaggle paths)
export HARMONIZER_DATA=/path/to/dir_with_final_corpus_and_lookup
export HARMONIZER_DATA1=/path/to/dir_with_extracted_json
export HARMONIZER_OUT=/path/to/output

# train one variant (rule switches are 0/1; defaults = full − {1,2,5})
ENABLE_RULES_1_2=1 ENABLE_RULE_5=1 MODEL_SEED=1 python src/train_harmonizer.py
```

Rule switches: `ENABLE_RULES_1_2` (rules 1 and 2), `ENABLE_RULE_3`, `ENABLE_RULE_5`, `ENABLE_RULE_6`, `ENABLE_RULE_7`, `ENABLE_RULE_8`. Examples: baseline = all `0`; full = all `1` (Rule 4 stays off). Training is 200 epochs, no early stopping; the last-epoch checkpoint is saved as `best_transformer_model.pth`.

Evaluation of all variants (accuracy, the 9 paper metrics, rule adherence, bootstrap CIs, figures):

```bash
export HARMONIZER_CKPTS=/path/to/final_data   # one sub-folder per variant, see VARIANTS in the script
python src/compare_ablation.py
```

`src/harmonize_musicxml.py` harmonizes a MusicXML melody with a trained checkpoint; `tools/` has converters for CMD/POP909 to MusicXML.

Note: the scripts were written as Kaggle notebooks, so they run top to bottom when executed and expect a GPU for training.

## Thesis

`thesis/` contains my original Thesis. Chapters are in Greek.

## Licence

Code: MIT (see `LICENSE`). The datasets keep their own licences.
