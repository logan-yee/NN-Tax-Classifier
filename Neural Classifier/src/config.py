"""Paths, hyperparameters, vocab, thresholds — single source of truth."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent  # "LightGBM Classifier/"

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

MASTER_XLSX = REPO_ROOT / "src" / "data" / "IncomeTax2025_MASTER_Christ&Jen.xlsx"
MASTER_SHEET = "MASTER 2025"
HEADER_ROW = 2  # pandas header= argument

TAX_EXTRACTOR_DIR = REPO_ROOT / "Tax Extractor" / "output"
EMAIL_EXTRACTOR_DIR = REPO_ROOT / "Email Extractor" / "output"

MASTER_LABELED_CSV = DATA_DIR / "master_labeled.csv"
INCOMING_CSV = DATA_DIR / "incoming.csv"
LABEL_MAPS_JSON = DATA_DIR / "label_maps.json"

STAGE1_DIR = MODELS_DIR / "stage1_category"
STAGE2_DIR = MODELS_DIR / "stage2_subcat"

TAX_YEAR = 2026
SEED = 42

SOURCE_VOCAB = ["BMO_CK", "BMO_LOC", "BMO_MC", "TD_Visa", "Tangerine", "CIBC"]

ENCODER_NAME = "distilbert-base-uncased"
MAX_LEN = 64
NUM_NUMERIC = 5  # log_amount, sign, month_sin, month_cos, is_weekend
HEAD_HIDDEN = 256
HEAD_DROPOUT = 0.3

ENCODER_LR = 2e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
EPOCHS = 5
BATCH_SIZE = 16
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 2
FREEZE_LOWER_LAYERS = 3
AUG_CHAR_NOISE_RATE = 0.03

CONFIDENCE_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Predict-time rule layer (additive — see src/rules.py).
# All matching is case-insensitive substring match on the normalized
# Description field (already cleaned by label_norm.clean_description).
# ---------------------------------------------------------------------------

# Vendor substring -> uppercase label written into the EXISTING Sub-Category2.
VENDOR_TERMS = {
    "paypal": "PAYPAL",
    "amazon": "AMAZON",
    "uber": "UBER",
    "temu": "TEMU",
}

# Car-related keywords -> Category=Business Expense, lean=business.
CAR_TERMS = [
    "car", "automotive", "auto ", "gas station", "fuel",
    "petro-canada", "petro canada", "esso", "shell", "mechanic",
    "oil change", "tire", "parking",
]

# Facebook / Meta keywords -> Category=Advertising, lean=business.
FACEBOOK_TERMS = ["facebook", "meta", "fb ads", "facebk"]

# Email-extractor contact substrings -> "expense" (business lean) or "personal".
# Contact text is folded into Description by ingest.build_desc, so these match
# against Description. Extend with real counterparties as they appear.
CONTACT_TERMS = {
    "shopify": "expense",
    "etsy": "expense",
}

# Reusable light-amber fill (openpyxl fgColor) for vendor-detected rows.
HIGHLIGHT_FILL_COLOR = "FFF2CC"
# Distinct red fill for UNRESOLVED rows (unmatched/unclassifiable e-transfers and
# low-confidence predictions) — visually separate from the amber vendor fill.
UNRESOLVED_FILL_COLOR = "FFC7CE"

# --- Interac e-transfer cross-reference -----------------------------------
# Transfer-keyword detection on the normalized Description.
TRANSFER_TERMS = ["interac", "e-transfer", "etransfer", "virement", "onlinetransfer", "online transfer"]

# Counterparty names ending in these tokens are treated as business contacts.
COMPANY_SUFFIXES = ["inc", "ltd", "ltee", "ltée", "corp", "incorporated", "llc", "enr", "srl"]

# Editable contact -> lean (business|personal) map. See load_known_contacts.
KNOWN_CONTACTS_CSV = DATA_DIR / "known_contacts.csv"

# Max |days| between a bank transfer and an email-extractor record to call it a match.
ETRANSFER_MATCH_WINDOW_DAYS = 7

# Smoke-test email-extractor source (used when EMAIL_EXTRACTOR_DIR has no interac CSV).
SMOKE_INTERAC_CSV = REPO_ROOT / "testdata_past4months" / "interac_transfers_2025-01-01_2026-05-14.csv"


def resolve_interac_sources() -> list:
    """Interac e-transfer CSV sources: real email-extractor output if present,
    else the checked-in smoke-test CSV."""
    real = sorted(EMAIL_EXTRACTOR_DIR.glob("*interac*transfer*.csv"))
    if real:
        return real
    return [SMOKE_INTERAC_CSV] if SMOKE_INTERAC_CSV.exists() else []


# Master workbook sheets to DROP when building the output copy (pivots/summaries
# that openpyxl cannot re-save reliably). All other sheets are preserved as-is,
# and a single "Predictions" sheet is appended.
DROP_SHEETS = ["P & L", "pivot Revenue", "pivot expenses HL", "pivot expenses detail"]
PREDICTIONS_SHEET = "Predictions"
