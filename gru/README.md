# GRU: individual contribution

Hotel review sentiment classification (SE4050 Deep Learning 2026). This folder holds the GRU model.

## Structure

```
gru/   (macOS treats GRU/ and gru/ as the same folder)
├── 01_Setup/                  GRU_Setup.ipynb
├── 02_Data_Audit/             GRU_Data_Audit.ipynb
├── 03_Dataset_and_Split/      GRU_Dataset_and_Split.ipynb
├── 04_EDA/                    GRU_EDA.ipynb
├── 05_Feature_Engineering/    GRU_Feature_Engineering.ipynb
├── 06_Class_Weights/          GRU_Class_Weights.ipynb
├── 07_GRU_Model/              GRU_Model.ipynb
├── 08_Training/               GRU_Training.ipynb
├── 09_Learning_Curves/        GRU_Learning_Curves.ipynb
├── 10_Final_Evaluation/       GRU_Final_Evaluation.ipynb
├── 11_Error_Analysis/         GRU_Error_Analysis.ipynb
├── 12_Robustness_Check/       GRU_Robustness_Check.ipynb
├── 13_Handover/               GRU_Handover.ipynb
├── 14_Optional_Extension/     GRU_Optional_Extension.ipynb   (also has the "try your own reviews" demo)
├── GRU_model.ipynb            the same 14 steps in ONE notebook (simplest for Colab)
├── results/                   figures, metrics, predictions, trained models
└── docs/                      report section and viva notes
```

Every notebook is **self-contained**: it defines the group settings and only the helper code it uses, and imports nothing from another file. The only input is the dataset `deceptive-opinion.csv`.

## Run it

**Locally (VS Code or Jupyter).** Use Python 3.10-3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Open the notebooks in order, `01_Setup` to `14_Optional_Extension`, and choose *Run All* in each. The dataset is found in `../dataset/`.

**Google Colab.** Colab opens every notebook in its own session, so for each notebook: upload it, choose *Runtime → Run all*, and pick `deceptive-opinion.csv` when it asks. Colab's disk is temporary, so download `results/` when you finish. Or upload just `GRU_model.ipynb`, which contains everything and needs one upload and one run.

**Any notebook can be run on its own.** Notebooks 09 to 14 need the trained GRU. They load it from `results/` if notebook 08 has already saved it there, and otherwise train it automatically first (about 15 seconds, same seed and settings, so the same result).

## What each notebook does and saves

| # | Notebook | Saves to `results/` |
|---|---|---|
| 01 | Libraries, dataset location, group settings, helper code | nothing |
| 02 | Data audit: missing values, duplicates, encoding, outliers | `gru_audit_length_outliers.png` |
| 03 | Labels and the fixed stratified 80/10/10 split | nothing |
| 04 | EDA: class balance, lengths, negation, distinctive words | `gru_eda_class_balance.png`, `gru_eda_distinctive_words.png` |
| 05 | Feature engineering: cleaning, vocabulary, ids, padding | `gru_eda_review_length.png` |
| 06 | Class weights | nothing |
| 07 | The GRU architecture and parameter count | nothing |
| 08 | Training with early stopping | `gru_model.keras`, `gru_history.csv`, `gru_training_info.json` |
| 09 | Learning curves | `gru_accuracy_curve.png`, `gru_loss_curve.png` |
| 10 | Final test evaluation (run once) | `gru_metrics.json`, `gru_test_predictions.csv`, `gru_confusion_matrix.png`, `gru_roc_curve.png`, `gru_probability_hist.png` |
| 11 | Error analysis (descriptive only) | nothing |
| 12 | Robustness across 5 other random seeds | `gru_seed_robustness.csv` |
| 13 | Row for the group comparison table, file list | nothing |
| 14 | Optional short-review extension (GRU only) and the demo | `gru_short_model.keras`, `gru_short_extension_metrics.json` |

Notebook `NN` here is the "Step NN-1" of the earlier all-in-one notebook.

## For the teammates (Simple RNN, LSTM, 1D CNN)

See [../TEAM_GUIDE.md](../TEAM_GUIDE.md) for the full guide. In short: copy the setup cells (settings, fixed validation/test row ids, data pipeline, metrics, plot style) unchanged from `01_Setup/GRU_Setup.ipynb` into your notebooks, then call `data = prepare_data()`. Only the model layers should differ. Save your results with `save_result(...)` so your `results/*.json` has the same format as `gru_metrics.json`.

The `../shared/` folder holds the same data pipeline as importable modules. The notebooks do not use it, but `../frontend/scripts/export_model.py` does (to rebuild the vocabulary for the web app), so keep it if you keep the frontend.

## Web app

The trained models can be tried in a browser: see [../frontend/README.md](../frontend/README.md).
