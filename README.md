# COMP60035 NLP Coursework (PCL Detection)

Simple overview:
- This repository is my full COMP60035 coursework submission for binary Patronising and Condescending Language (PCL) detection.
- The key assessment files are at the root (`dev.txt`, `test.txt`), in `BestModel/` (final training code + model weights), and in `report/` (final PDF report).
- The sections below give direct clickable links so markers can quickly open the exact files used for the final submission and results.

This repository contains the full submission for COMP60035 Natural Language Processing coursework (SemEval-2022 Task 4, Subtask 1: binary PCL detection).

## Quick Marker Navigation

### Core submission outputs
- Dev predictions (root): [dev.txt](dev.txt)
- Test predictions (root): [test.txt](test.txt)

### Best model implementation
- Best model training code: [BestModel/best_model_train.py](BestModel/best_model_train.py)

### Best model weights (selected ensemble members)
The final system is a weighted ensemble, not a single checkpoint: we trained RoBERTa and DeBERTa across multiple seeds, then selected the strongest runs by validation F1.  
These linked weight files are the exact selected members whose probabilities were combined and threshold-tuned to produce the final `dev.txt`/`test.txt`.

- DeBERTa-v3 (seed 1337): [BestModel/model_weights/microsoft_deberta-v3-base__seed_1337/model.safetensors](BestModel/model_weights/microsoft_deberta-v3-base__seed_1337/model.safetensors)
- DeBERTa-v3 (seed 627345): [BestModel/model_weights/microsoft_deberta-v3-base__seed_627345/model.safetensors](BestModel/model_weights/microsoft_deberta-v3-base__seed_627345/model.safetensors)
- DeBERTa-v3 (seed 42): [BestModel/model_weights/microsoft_deberta-v3-base__seed_42/model.safetensors](BestModel/model_weights/microsoft_deberta-v3-base__seed_42/model.safetensors)
- RoBERTa-base (seed 42): [BestModel/model_weights/roberta-base__seed_42/model.safetensors](BestModel/model_weights/roberta-base__seed_42/model.safetensors)

### EDA, baseline, evaluation artifacts
- Baseline Python file: [baseline/baseline_roberta_official.py](baseline/baseline_roberta_official.py)
- Error-analysis/evaluation artifacts: [evaluation/](evaluation/)
  - Full dev error dataset: [evaluation/dev_error_analysis_dataset.csv](evaluation/dev_error_analysis_dataset.csv)
  - Category FN analysis: [evaluation/dev_error_category_fn.csv](evaluation/dev_error_category_fn.csv)
  - Error hotspots: [evaluation/dev_error_hotspots.csv](evaluation/dev_error_hotspots.csv)
  - Keyword-level metrics: [evaluation/dev_error_keyword_metrics.csv](evaluation/dev_error_keyword_metrics.csv)
  - Length-bin metrics: [evaluation/dev_error_lengthbin_metrics.csv](evaluation/dev_error_lengthbin_metrics.csv)
  - Sample false negatives: [evaluation/dev_error_samples_fn.csv](evaluation/dev_error_samples_fn.csv)
  - Sample false positives: [evaluation/dev_error_samples_fp.csv](evaluation/dev_error_samples_fp.csv)

### Spec, report, and notes
- Coursework report (PDF): [report/Adithya_Narayanan_NLP_Coursework_2026.pdf](report/Adithya_Narayanan_NLP_Coursework_2026.pdf)
- Coursework report source (LaTeX): [report/Adithya Narayanan, Natural Language Processing Coursework, 2026.tex](<report/Adithya Narayanan, Natural Language Processing Coursework, 2026.tex>)

## Repository Structure (Tree)

```text
.
├── BestModel/
│   ├── best_model_train.py
│   ├── dev.txt
│   ├── test.txt
│   └── model_weights/
│       ├── microsoft_deberta-v3-base__seed_1337/
│       ├── microsoft_deberta-v3-base__seed_627345/
│       ├── microsoft_deberta-v3-base__seed_42/
│       └── roberta-base__seed_42/
├── baseline/
│   └── baseline_roberta_official.py
├── data/
│   ├── dontpatronizeme_pcl.tsv
│   ├── task4_test.tsv
│   └── practice splits/
├── eda/
│   └── appendix_b_full_eda.ipynb
├── evaluation/
│   ├── dev_error_analysis_dataset.csv
│   ├── dev_error_category_fn.csv
│   ├── dev_error_hotspots.csv
│   ├── dev_error_keyword_metrics.csv
│   ├── dev_error_lengthbin_metrics.csv
│   ├── dev_error_samples_fn.csv
│   └── dev_error_samples_fp.csv
├── literature/
│   ├── 60035_1_spec.pdf
│   └── 2020.coling-main.518.pdf
├── report/
│   ├── Adithya_Narayanan_NLP_Coursework_2026.pdf
│   ├── Adithya Narayanan, Natural Language Processing Coursework, 2026.tex
│   └── figures/
├── .bestmodel_runs_ensemble/
│   └── summary.json
├── .notes.md
├── dev.txt
└── test.txt
```

## Final Results Table (from report / run summary)

| System / Run | Scout val F1 | Scout dev F1 | Final dev F1 | Threshold |
|---|---:|---:|---:|---:|
| DeBERTa-v3 (seed=1337) | 0.6817 | 0.5907 | 0.6076 | 0.945 |
| DeBERTa-v3 (seed=627345) | 0.6775 | 0.5763 | 0.6146 | 0.910 |
| DeBERTa-v3 (seed=42) | 0.6426 | 0.5876 | 0.5979 | 0.935 |
| RoBERTa-base (seed=42) | 0.6343 | 0.5714 | 0.5714 | 0.665 |
| Scout weighted ensemble | 0.6772 | 0.6071 | -- | 0.485 |
| Final ensemble (pre-retune; selection threshold) | -- | -- | 0.6037 | 0.485 |
| **Final ensemble (retuned on dev)** | -- | -- | **0.6336** | **0.250** |

## Baseline Comparison on Dev (Official baseline F1 = 0.4800)

| Model type | Final dev F1 | Baseline dev F1 | Absolute improvement | Relative improvement |
|---|---:|---:|---:|---:|
| RoBERTa-base (single model) | 0.5714 | 0.4800 | +0.0914 | +19.0% |
| DeBERTa-v3 (best single seed) | 0.6146 | 0.4800 | +0.1346 | +28.0% |
| **Final weighted ensemble (retuned)** | **0.6336** | 0.4800 | **+0.1536** | **+32.0%** |

## Submission Format Checks

- `dev.txt` line count: 2094 (one prediction per line, values in `{0,1}`)
- `test.txt` line count: 3832 (one prediction per line, values in `{0,1}`)
- Root and `BestModel/` copies are identical for both files.
