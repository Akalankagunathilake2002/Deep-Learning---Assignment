# Team guide: how to build your model so the four models compare fairly

For the people building the **Simple RNN**, **LSTM** and **1D CNN** parts. The GRU part in [`gru/`](gru/) is the reference. If you follow it, the four results can go straight into one comparison table, and the comparison is fair because **only the model architecture differs**.

---

## 1. What has to be confirmed by the whole group

The GRU part made these choices so it could get started. Please check them together, and if any changes, **all four models must be re-run**.

| Decision | What the GRU part uses | Why |
|---|---|---|
| Dataset | `dataset/deceptive-opinion.csv` (1,600 hotel reviews, 20 Chicago hotels) | It is the file in the repo |
| Label | The `polarity` column: **positive = 1, negative = 0** | This file has **no star ratings**, so the "rating ≥ 4 / < 3" rule cannot be applied, and there are no neutral reviews to remove |
| Shared settings | Section 2 below | Proposed by the GRU part, nothing else existed yet |
| Folder names | The GRU folder is `gru/` (lowercase) | On Mac and Windows `GRU/` and `gru/` are the same folder, but on Linux and Colab they are different. Pick one spelling for all four (for example `gru/`, `lstm/`, `simple_rnn/`, `cnn1d/`) **before the first commit** |

---

## 2. What must be identical in every model (copy it, do not retype it)

Copy the five helper cells from [`gru/01_Setup/GRU_Setup.ipynb`](gru/01_Setup/GRU_Setup.ipynb) **unchanged** into your notebooks (`shared/` holds the same code as Python files). They contain the settings, the fixed split, the data pipeline, the metrics and the plot style. Then call `data = prepare_data()`.

| Item | Value |
|---|---|
| Random seed | 42 |
| Duplicates | 4 exact duplicate reviews are removed **before** splitting (1,600 → 1,596) |
| Split | Stratified 80 / 10 / 10: **1,276 train, 160 validation, 160 test**. It is fixed by the `VAL_IDS` and `TEST_IDS` lists in the Settings cell. **Never generate your own split** |
| Cleaning | `clean_text()`: lowercase; remove HTML, URLs, digits and punctuation; keep negations ("didn't" → "did not"); keep stop-words |
| Vocabulary | 5,000 ids (`<PAD>` = 0, `<OOV>` = 1), built from the **training set only** |
| Sequence length | `MAX_LEN` = 300, zeros in front (`padding="pre"`), long reviews cut at the end (`truncating="post"`) |
| Embedding | `Embedding(5000, 64)`, learned from scratch, no pretrained vectors |
| Class weights | `n / (n_classes × n_in_class)` from the training labels. The classes are balanced here (640 / 636), so the weights are about 1.0. Say so honestly in your report |
| Optimizer / loss | Adam, learning rate 1e-3 / binary cross-entropy |
| Batch size / max epochs | 32 / 30 |
| EarlyStopping | monitor `val_loss`, patience 5, `restore_best_weights=True` |
| Decision threshold | 0.5 |
| Metrics | accuracy, precision, recall, F1 (all for the **positive** class), ROC-AUC (from the probabilities), confusion matrix. Use `compute_metrics()` so everyone calculates them the same way |

## 3. What you are free to change

Only the layers between the Embedding and the output. The GRU model is:

```
Input(300) → Embedding(5000, 64) → GRU(64) → Dropout(0.5) → Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
```

- **Simple RNN:** replace `GRU(64)` with `SimpleRNN(64)`.
- **LSTM:** replace `GRU(64)` with `LSTM(64)`.
- **1D CNN:** replace it with your own feature extractor, for example `Conv1D(...)` then `GlobalMaxPooling1D()`, and explain your choice. Keep the same Embedding, the same head and the same training settings.

Keep the rest of the head the same unless the group agrees otherwise. Do not use `mask_zero` (Conv1D cannot use it, so leaving it off keeps the comparison fair). Do not tune many hyperparameters: if you change anything beyond the architecture, tell the group.

---

## 4. Folder and notebook structure to follow

```
gru/
├── 01_Setup/                GRU_Setup.ipynb                libraries, dataset, settings, helper code
├── 02_Data_Audit/           GRU_Data_Audit.ipynb           missing values, duplicates, encoding, outliers
├── 03_Dataset_and_Split/    GRU_Dataset_and_Split.ipynb    labels and the 80/10/10 split
├── 04_EDA/                  GRU_EDA.ipynb                  exploratory analysis (training set)
├── 05_Feature_Engineering/  GRU_Feature_Engineering.ipynb  cleaning, vocabulary, ids, padding
├── 06_Class_Weights/        GRU_Class_Weights.ipynb
├── 07_GRU_Model/            GRU_Model.ipynb                architecture and parameter count      <- changes
├── 08_Training/             GRU_Training.ipynb             training with early stopping          <- changes
├── 09_Learning_Curves/      GRU_Learning_Curves.ipynb
├── 10_Final_Evaluation/     GRU_Final_Evaluation.ipynb     the ONE test-set evaluation
├── 11_Error_Analysis/       GRU_Error_Analysis.ipynb
├── 12_Robustness_Check/     GRU_Robustness_Check.ipynb     5 other random seeds
├── 13_Handover/             GRU_Handover.ipynb             row for the comparison table
├── 14_Optional_Extension/   GRU_Optional_Extension.ipynb   optional, GRU only
├── results/                 all outputs
└── docs/                    report section and viva notes
```

The quickest way: **copy the whole `gru/` folder, rename it, rename the notebooks and the `gru_` file prefix to your model's name, then edit notebooks 07 and 08** (plus the wording in the others). Notebooks 01 to 06 do not depend on the model, so they will be the same for everyone.

Rules each notebook follows, and yours should too:

- **Self-contained.** It includes the settings and the helper code it uses and imports nothing from another file, so it runs on its own, in order, and in Colab (upload the notebook, pick `deceptive-opinion.csv` when it asks).
- **Runs top to bottom** with *Run All* and no errors, and finds the dataset in `../dataset/`.
- **Passes the trained model on through `results/`:** notebook 08 saves the model, and later notebooks load it (or train it automatically if it is missing). See [`gru/README.md`](gru/README.md).

---

## 5. Files your part must produce

Use your model name as the prefix (`lstm_`, `simple_rnn_`, `cnn1d_`) and save them in your own `results/` folder. The GRU files are the examples.

| File | Created by | Why it matters |
|---|---|---|
| `<model>_metrics.json` | `save_result(...)` in notebook 10 | The row for the group table. **Same keys for every model** (below) |
| `<model>_test_predictions.csv` | notebook 10 | Columns `row_id, label, prob_positive, pred`. Lets the group compare models review by review (for example McNemar's test) |
| `<model>_history.csv` | notebook 08 | Training and validation loss and accuracy per epoch |
| `<model>_accuracy_curve.png`, `<model>_loss_curve.png` | notebook 09 | Required figures |
| `<model>_confusion_matrix.png`, `<model>_roc_curve.png` | notebook 10 | Required figures. Use the same `plot_*` helpers so all four look alike |
| `<model>_model.keras` | notebook 08 | The trained model |
| `<model>_seed_robustness.csv` | notebook 12 | Optional: how much the result moves between seeds |

Keys in `<model>_metrics.json`: `model, accuracy, precision, recall, f1, roc_auc, training_time_sec, trainable_params, total_params, epochs_run, best_epoch, seed, max_len, vocab_size, embed_dim, batch_size, learning_rate, threshold, confusion_matrix`. Call `save_result(path, "LSTM", test_metrics, training_time_sec=..., trainable_params=..., total_params=..., epochs_run=..., best_epoch=...)`.

The final table has one row per model:

`Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Training Time | Parameters`

---

## 6. The rules that keep the comparison honest

- The vocabulary and the class weights come from the **training set only**.
- Remove duplicates **before** splitting, and never regenerate the split.
- The **validation** set is for EarlyStopping and every development decision.
- The **test** set is used **once per model**, at the end (notebook 10). Never use it to tune, to choose a model, to choose an epoch or to set a threshold.
- Error analysis and the seed check are **descriptive only**: nothing changes after you see them.
- Your headline result is the **seed 42** run, fixed in advance. Other seeds only show the variation (report mean ± std). Do not pick the best seed.
- Measure **training time on the same hardware** for all four models. The GRU's is about 14 s on an Apple M4 CPU. Colab times will differ, so agree where to time it.
- Do **not** name a best model until all four have been evaluated under these conditions.

## 7. What the GRU work found that may matter for your model

- **The test set is small (160 reviews).** The GRU's 88.1% accuracy has a 95% range of 83.1% to 92.5%. Differences of a few points between models are within noise, so say that in the comparison. Use the predictions files and the seed spread to back it up.
- **Length is a weak shortcut.** Negative reviews are longer (mean 180 vs 119 words), and a rule using length alone gets 67.5% on validation. With pre-padding a recurrent model can see how long a review is.
- **Very short reviews.** Every training review has at least 25 words. The GRU labelled all 16 negative one-line reviews it was given as positive, so it is worth trying a few one-liners on your model too (notebook 14 has a list). Do not assume your model behaves the same way. If the group wants to fix this, the fix must be applied to all four models.
- **Over-confidence.** The GRU often gives more than 90% confidence to wrong answers.
- **It overfits fast.** The GRU peaked at epoch 5 and stopped at epoch 10, so keep EarlyStopping.
- **No class imbalance in this file** (800 positive, 800 negative). Class weights are about 1.0.

## 8. Report and viva

- Write your report section with the same headings as [`gru/docs/GRU_report_section.md`](gru/docs/GRU_report_section.md): role in the study, data and preprocessing, architecture, training configuration, training results and curves, final test evaluation, error analysis, model analysis, strengths and limitations, your row for the group table, critical analysis. Data and preprocessing are shared, so the group can describe them once.
- Take every number from **your** `results/<model>_metrics.json`, and only claim what your results show.
- For viva preparation, [`gru/docs/GRU_theory_viva.md`](gru/docs/GRU_theory_viva.md) is a template: what the model is, how it processes a sequence, how it differs from the others, advantages and limits, and likely questions.

## 9. Checklist before you hand in

- [ ] The setup cells are copied unchanged from `gru/01_Setup/GRU_Setup.ipynb`
- [ ] The split is 1,276 / 160 / 160 and uses the fixed `VAL_IDS` and `TEST_IDS`
- [ ] Only the model layers differ from the GRU notebooks
- [ ] Every notebook runs top to bottom without errors (also in Colab)
- [ ] The test set is used once, in notebook 10
- [ ] All the result files in section 5 exist, with the same keys in the metrics JSON
- [ ] The report numbers come from the run you are reporting
- [ ] Training time was measured on the agreed hardware

## 10. The web app

[`frontend/`](frontend/) is a React page that runs the two GRU models in the browser. It only knows the GRU maths. Adding another model type would need its own forward pass, so talk to the GRU owner first if you want yours in it.
