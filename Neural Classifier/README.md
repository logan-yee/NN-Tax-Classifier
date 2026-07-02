# Neural Classifier — Tax Categorization

Fine-tuned DistilBERT two-stage classifier for bank/credit-card transactions.

- **Stage 1** predicts `Category` ∈ {EXPENSES, PERSONAL, REVENUE}.
- **Stage 2** predicts `Sub-Category1` for EXPENSES rows only.

Inputs come from sibling folders: `../Tax Extractor/output/` (PDF→CSV per bank) and `../Email Extractor/output/` (IMAP→CSV for Interac/invoices/orders). 

## Setup

```powershell
cd "Filepath"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
# one-time, when MASTER xlsx changes
python -m src.build_training

# each cycle
python -m src.ingest         # extractor CSVs -> data/incoming.csv
python -m src.train --stage 1
python -m src.train --stage 2
python -m src.predict        # -> output/classified_<date>.xlsx
python -m src.evaluate --stage 1   # optional, prints confusion matrix
```

## Layout

```
src/
  config.py        paths, hparams, Source vocab
  schema.py        MASTER 19-col list
  label_norm.py    EXCLUDE_CATEGORIES + label maps
  ingest.py        extractor CSVs -> MASTER rows
  build_training.py MASTER xlsx -> training CSV + label maps
  dataset.py       torch Dataset + tokenizer + numeric features
  model.py         DistilBertWithFeatures
  train.py         HF Trainer wrapper, --stage CLI
  predict.py       two-stage inference + xlsx writer
  evaluate.py      confusion matrix + per-class report
```

