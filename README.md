# WireCF v3 — Wire-Valid Counterfactual Flows for Encrypted Traffic Classifiers

Publication-grade implementation of a counterfactual-explanations framework for encrypted-traffic classifiers, with parameter-matched Transformer / Mamba / xLSTM backbones, masked-byte pretraining, wire-valid DDPM-based counterfactuals, Integrated-Gradients causal-faithfulness scoring, and an end-to-end honesty contract with per-stage convergence gates.


## Repository layout

```
wirecf-v3/
├── src/wirecf/                  # Python package
│   ├── config.py                # CFG dict, seeding, audit logger, STAGE_STATUS gates
│   ├── data.py                  # FlowDataset with anti-synthetic guard, Kaggle multi-mirror loader
│   ├── models/
│   │   ├── tokenizer.py         # PatchTokenizer (YaTC 3-channel byte matrix)
│   │   ├── transformer.py       # TransformerBackbone
│   │   ├── mamba.py             # S6Block + MambaBackbone (PyTorch-only, portable)
│   │   ├── xlstm.py             # MLSTMCell (dim-scaled) + SLSTMCell + XLSTMBackbone
│   │   ├── attention.py         # InterFlowAttention (audit target)
│   │   ├── classifier.py        # WireCFClassifier (pretrain + classification heads)
│   │   └── param_match.py       # auto_match_params, build_matched_models
│   ├── train/
│   │   ├── pretrain.py          # Masked-byte pretraining
│   │   └── finetune.py          # Fine-tune loop, convergence gate, majority baseline
│   ├── cf/
│   │   ├── ddpm.py              # TinyUNet + DDPM_CF with edit-mask
│   │   └── wire_valid.py        # Protocol-aware projection + wire-validity check
│   ├── xai/
│   │   ├── ig.py                # Integrated Gradients + CFS
│   │   └── battery.py           # FVT + ERASER (sufficiency, comprehensiveness)
│   ├── shortcut/obfuscation.py  # 8 factorial shortcut obfuscation operators
│   ├── stats.py                 # Paired bootstrap CI, Holm–Bonferroni
│   └── evaluation.py            # Auto-eval H1..H5 with gating
├── scripts/
│   ├── run_all.py               # End-to-end pipeline
│   ├── download_data.py         # Kaggle mirror chain (standalone)
│   └── generate_report.py       # Final honest report + reproducibility bundle
├── notebooks/
│   └── WireCF_v3_publication_notebook.ipynb   # Original Colab notebook
├── configs/default.yaml         # Externalized configuration
├── docs/                        # Architecture, hypotheses, v2→v3 changes
├── tests/                       # Smoke tests
├── results/                     # Populated at runtime (CSVs, JSON, plots)
└── ckpts/                       # Populated at runtime (model checkpoints)
```

---

## Installation

```bash
git clone https://github.com/amrogomma-web/wirecf-v3.git
cd wirecf-v3
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
pip install -e .
```

Datasets are downloaded from Kaggle. Place your `kaggle.json` API token at `~/.kaggle/kaggle.json` (mode 600) before running the pipeline.

---

## Quick start

### Option A — run the notebook end-to-end (recommended for first pass)

Open `notebooks/WireCF_v3_publication_notebook.ipynb` in Colab, select an A100 runtime, upload `kaggle.json` when prompted, and run all cells top-to-bottom. The final cell (§14) prints the honest report.

### Option B — run the modular pipeline from CLI

```bash
# 1. Fetch datasets (tries mirror chain per dataset)
python scripts/download_data.py --config configs/default.yaml

# 2. Full pipeline: param match → pretrain → fine-tune → shortcut sweep → CF → CFS → battery → hypotheses
python scripts/run_all.py --config configs/default.yaml

# 3. Print the final honest report + write the reproducibility bundle
python scripts/generate_report.py
```

For a fast code-review pass on a T4 (skips paper-mode gates), set `FULL_RUN: false` in the config.

---

## Pre-registered hypotheses

Full statements are in [`docs/HYPOTHESES.md`](docs/HYPOTHESES.md). Summary:

- **H1** — Wire-validity of CFs ≥ 0.60.
- **H2** — CFS and FVT disagree by ≥ 0.20 on at least one (backbone, dataset).
- **H3** — No backbone is uniformly faithful across all datasets (at most 2 passes).
- **H4** — The highest-F1 backbone has ≥ 3 non-robust singleton features.
- **H5** — Variance across backbones < variance across shortcut features.

Every hypothesis is auto-evaluated in §12 of the pipeline. Each one is marked either `evaluated` (with `supported=True/False`) or `UNINTERPRETABLE — <upstream stage failed>`. **Uninterpretable is never converted to supported.**

---

## Reproducibility

After a run, `scripts/generate_report.py` produces `results/wirecf_v3_reproducibility_bundle.zip` containing:

- All CSV results (`shortcut_sweep.csv`, `cfs.csv`, `faithfulness_battery.csv`)
- Hypothesis evaluation (`pre_reg_evaluation.json`)
- Config snapshot and stage-status flags
- Full JSONL audit log from `AuditLog`
- Pre-registration record (timestamp + decision rules)

Five seeds (13, 42, 101, 2024, 31337) are used for paper-grade statistics; three are used by default in the fine-tune stage.

---

## Citation

If you use this code, please cite:

```bibtex
@article{abdelhaliem2025wirecf,
  title   = {WireCF: Wire-Valid Counterfactual Explanations for Encrypted Traffic Classifiers},
  author  = {Abdelhaliem, Amr H. and others},
  journal = {IEEE Transactions on Information Forensics and Security (submitted)},
  year    = {2025}
}
```

See [`CITATION.cff`](CITATION.cff) for a machine-readable citation.

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

---

## Acknowledgements

Datasets: CIC-IoT2022, ISCXVPN2016 (Canadian Institute for Cybersecurity), USTC-TFC2016. Backbone references: PyTorch, `xlstm` package, `mamba-ssm` community implementations.
