# NN Tax Classifier

An end-to-end pipeline that turns a year's worth of bank/credit-card statements
and transaction emails into a categorized, tax-ready Excel workbook. Raw
statements and inbox notifications go in; a `Predictions` sheet with a predicted
`Category` / `Sub-Category1` per transaction (plus review highlighting) comes out.

The project is three loosely coupled components that hand CSVs to each other:

```
  PDF statements ──▶ Tax Extractor ────┐
                                       ├──▶ Neural Classifier ──▶ classified_<date>.xlsx
  Inbox (IMAP)   ──▶ Email Extractor ──┘
```

There is **no single orchestrator** wiring these together — each stage is run by
hand, and the classifier's `ingest` step is what globs the two extractors'
`output/` folders and merges them. This is intentional; the extractor output is a
human-in-the-loop input that gets reviewed before classification.

---

## Components

### 1. `Tax Extractor/` — PDF statements → CSV

A Jupyter notebook (`notebooks/extractor.ipynb`) that parses bank / credit-card
statement PDFs into per-bank CSVs of transactions. Each bank (BMO LOC, BMO MC,
BMO Chequing, Tangerine, TD) has its own regex and a dedicated
`extract_<bank>_folder_to_csv` function. Run it by editing the
`root_<bank>_path` / `output_csv` constants in the `__main__` cell and executing.

- **Output:** `Tax Extractor/output/*.csv` (`BMO_CK_transactions.csv`,
  `BMO_LOC_transactions.csv`, `BMO_MC_transactions.csv`,
  `Tangerine_transactions.csv`, `TD_transactions.csv`).
- Columns: `TRANS DATE`, `POSTING DATE` (some banks), `DESCRIPTION`, `AMOUNT`,
  `CARD TYPE` (debit/credit).

### 2. `Email Extractor/` — inbox → CSV

`email_extractor.py` logs into an IMAP mailbox **read-only** and scrapes three
kinds of messages within a date range: Interac e-Transfer notifications,
invoices/receipts, and online-order confirmations. Each is parsed by a dedicated,
side-effect-free parser (`interac_parser.py`, `invoice_parser.py`,
`order_parser.py`) and written to a single combined CSV + XLSX.

- **Run:** `python email_extractor.py` (interactive — prompts for address, IMAP
  host, app password, and date range).
- **Output:** `Email Extractor/output/email_records_<since>_<until>.csv` (+ `.xlsx`).
- A `TYPE` column tags each row `transfer` / `invoice` / `order`.

**Security posture (non-negotiable):** password read via `getpass` and never
stored; `IMAP4_SSL` on port 993 only; `INBOX` opened `readonly=True` and fetched
with `BODY.PEEK[]` so nothing is marked seen; server-side `SEARCH` narrows before
any body download; no outbound HTTP (hosted invoice URLs are recorded, never
fetched); attached PDFs are parsed in memory and discarded.

- **Optional deps** (invoice OCR fallback): `pdfplumber`, `pdf2image` (needs the
  Poppler binary on PATH), `pytesseract` (needs the Tesseract binary on PATH),
  `Pillow`. If Poppler/Tesseract are missing the run does **not** crash — scanned
  PDFs simply yield no row. `openpyxl` is required for the XLSX writer. See
  `Email Extractor/requirements.txt`.

### 3. `Neural Classifier/` — CSVs → classified workbook

A fine-tuned **DistilBERT** two-stage classifier. This is the main codebase; it
has its own detailed [`Neural Classifier/README.md`](Neural%20Classifier/README.md).

- **Stage 1** predicts `Category` ∈ {EXPENSES, PERSONAL, REVENUE}.
- **Stage 2** predicts `Sub-Category1`, on Stage-1 EXPENSES rows only.
- Each stage is `DistilBertWithFeatures`: the encoder's `[CLS]` embedding of the
  cleaned `Description` is concatenated with 5 numeric features (signed
  `log_amount`, sign, month sin/cos, is-weekend) and a one-hot `Source`, then fed
  through an MLP head.
- A **predict-time rule layer** (`src/rules.py`) sits on top of the model output
  without retraining it: vendor keyword tags (PayPal/Amazon/Uber/Temu → written
  into `Sub-Category2`), source rules (all TD → Advertising), car/Facebook keyword
  categories, BMO/Tangerine business-vs-personal lean, and an **Interac
  e-transfer cross-reference** that recovers a transfer's counterparty by matching
  the bank line against Email Extractor records (date ± window + amount +
  direction). A `Decision_Source` column records which layer decided each row.

---

## Running the full pipeline

Everything below runs from `Neural Classifier/`, which globs the two sibling
extractor `output/` folders.

```powershell
# 0. Produce extractor CSVs first
#    - run Tax Extractor/notebooks/extractor.ipynb  -> Tax Extractor/output/*.csv
#    - run  python Email Extractor/email_extractor.py -> Email Extractor/output/*.csv

cd "Neural Classifier"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. One-time (and whenever the labeled MASTER xlsx changes): build training data
python -m src.build_training      # MASTER 2025 xlsx -> data/master_labeled.csv + label_maps.json

# 2. Train the two stages (writes models/stage1_category/ and models/stage2_subcat/)
python -m src.train --stage 1
python -m src.train --stage 2

# 3. Each classification cycle
python -m src.ingest              # extractor CSVs -> data/incoming.csv (mapped to 19-col MASTER schema, deduped)
python -m src.predict             # -> output/classified_<date>.xlsx

# Optional
python -m src.evaluate --stage 1  # held-out confusion matrix + per-class report
```

The output workbook is a copy of the master with pivot/summary sheets dropped and
a single `Predictions` sheet appended. Rows are highlighted for review: **amber**
for vendor-detected rows, **red** for unresolved e-transfers and low-confidence
predictions (below `CONFIDENCE_THRESHOLD = 0.70`).

## Configuration

`Neural Classifier/src/config.py` is the single source of truth for paths,
hyperparameters, the `Source` vocabulary, confidence threshold, and the rule-layer
keyword/vendor/contact tables. Editable data files:

- `data/known_contacts.csv` — contact → `business`/`personal` lean for the
  e-transfer cross-reference (supports `#` comment lines).
- `data/label_maps.json` — the class lists emitted by `build_training`.

Label normalization (historical spelling/casing fixes, excluded categories like
`TRANSFER`/`BORROWING`) lives in `src/label_norm.py`. Add new inconsistencies
found in future years there.

## Repository layout

```
NN Tax Classifier/
├── Tax Extractor/
│   ├── notebooks/extractor.ipynb   PDF statements -> per-bank CSVs
│   └── output/                     *_transactions.csv
├── Email Extractor/
│   ├── email_extractor.py          IMAP orchestrator (run this)
│   ├── interac_parser.py           Interac e-transfer emails
│   ├── invoice_parser.py           invoice/receipt emails (PDF + OCR fallback)
│   ├── order_parser.py             online-order confirmations
│   ├── email_utils.py
│   └── output/                     email_records_<since>_<until>.csv / .xlsx
└── Neural Classifier/
    ├── src/
    │   ├── config.py               paths, hparams, Source vocab, rule tables
    │   ├── schema.py               MASTER 19-column schema
    │   ├── label_norm.py           exclusions + label maps + description cleaner
    │   ├── ingest.py               extractor CSVs -> data/incoming.csv
    │   ├── build_training.py       MASTER xlsx -> training CSV + label maps
    │   ├── dataset.py              torch Dataset, tokenizer, numeric features
    │   ├── model.py                DistilBertWithFeatures (encoder + numeric -> MLP head)
    │   ├── train.py                per-stage training (--stage 1|2)
    │   ├── predict.py              two-stage inference + xlsx writer
    │   ├── rules.py                additive predict-time rule layer
    │   ├── evaluate.py             confusion matrix + per-class report
    │   └── test_rules.py           unit tests for the rule layer
    ├── data/                       incoming.csv, master_labeled.csv, *.json/csv
    ├── models/                     stage1_category/, stage2_subcat/
    └── output/                     classified_<date>.xlsx
```

## Requirements

Python 3.10+. Core: `torch`, `transformers`, `pandas`, `numpy`, `scikit-learn`,
`openpyxl` (see `Neural Classifier/requirements.txt`). The DistilBERT encoder
(`distilbert-base-uncased`) is downloaded from Hugging Face on first run.

## Notes

- The classifier trains on the labeled `MASTER 2025` sheet of the tax workbook
  referenced by `config.MASTER_XLSX`; `TRANSFER` and `BORROWING` rows are excluded
  from training because they aren't tax-relevant and add label noise.
- There are no linters or a build system — the extractors and classifier are
  script/notebook pipelines. `Neural Classifier/src/test_rules.py` covers the rule
  layer (`python -m pytest src/test_rules.py`).
- `Step-by-step guide.mp4` in the project root is a screen-recorded walkthrough.
```
