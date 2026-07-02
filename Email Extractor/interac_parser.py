"""
interac_parser.py — Parse Interac e-Transfer notification emails.

Pure, side-effect free. Operates on `email.message.Message` objects and
returns a normalized `InteracRecord` (or `None` for messages that don't
match a known format).

Dispatch order in `parse_interac_email`:
  1. Filter on subject — skips requests / cancellations / declines / expiries.
  2. Route by `From:` domain to a per-format sub-parser.
  3. Sub-parsers delegate to `_parse_generic`, which is the shared
     amount / contact / account / reference / memo / date extractor.

Per-bank sub-parsers are intentionally thin wrappers today. The seam is
there so a future contributor can specialize one without rewriting the
dispatcher.
"""

import datetime
import fnmatch
import re
from dataclasses import dataclass
from decimal import Decimal
from email.message import Message
from email.utils import parseaddr
from typing import Callable, Optional

from email_utils import (
    email_received_date as _email_received_date,
    get_body_text as _get_body_text,
    parse_amount as _parse_amount,
    parse_date as _parse_date,
)


ALLOWED_SENDERS = [
    "notify@payments.interac.ca",
    "catch@payments.interac.ca",
    "*@payments.interac.ca",
    "*@interac.ca",
    "*@bmo.com",
    "*@td.com",
    "*@tdcanadatrust.com",
    "*@tangerine.ca",
]


CSV_COLUMNS = [
    "DATE", "DIRECTION", "AMOUNT", "BANK", "CONTACT",
    "ACCOUNT", "REFERENCE", "MEMO", "SUBJECT",
]


@dataclass
class InteracRecord:
    date: datetime.date
    direction: str            # "sent" | "received"
    amount: Decimal
    bank: str                 # canonical bank label, e.g. "BMO", "TD"; "" if unknown
    contact: str
    account: str
    reference: str
    memo: str
    subject: str

    def to_csv_row(self) -> dict:
        return {
            "DATE": self.date.isoformat(),
            "DIRECTION": self.direction,
            "AMOUNT": f"{self.amount:.2f}",
            "BANK": self.bank,
            "CONTACT": self.contact,
            "ACCOUNT": self.account,
            "REFERENCE": self.reference,
            "MEMO": self.memo,
            "SUBJECT": self.subject,
        }

    def to_excel_row(self) -> dict:
        """Native-typed row for openpyxl (date object, float amount)."""
        return {
            "DATE": self.date,
            "DIRECTION": self.direction,
            "AMOUNT": float(self.amount),
            "BANK": self.bank,
            "CONTACT": self.contact,
            "ACCOUNT": self.account,
            "REFERENCE": self.reference,
            "MEMO": self.memo,
            "SUBJECT": self.subject,
        }


# ---
# Public helpers
# ---

def sender_matches(addr: str, allowlist: list[str]) -> bool:
    """Glob-match a single email address against an allowlist of patterns."""
    addr = (addr or "").lower().strip()
    if not addr:
        return False
    return any(fnmatch.fnmatch(addr, pat.lower()) for pat in allowlist)


def parse_amount_filter(s: str) -> Callable[[Decimal], bool]:
    """
    Build a predicate from strings like '>=500', '<=20', '=1000', '>0', or ''.
    Empty input returns a predicate that always returns True.
    """
    s = (s or "").strip()
    if not s:
        return lambda _amount: True
    m = re.match(r"^(>=|<=|=|>|<)\s*(\d+(?:\.\d+)?)$", s)
    if not m:
        raise ValueError(
            f"Invalid amount filter: {s!r}. Expected forms: >=500, <=20, =1000, >0."
        )
    op, val = m.group(1), Decimal(m.group(2))
    return {
        ">=": lambda x: x >= val,
        "<=": lambda x: x <= val,
        "=":  lambda x: x == val,
        ">":  lambda x: x > val,
        "<":  lambda x: x < val,
    }[op]


# ---
# Dispatcher
# ---

def parse_interac_email(msg: Message) -> Optional[InteracRecord]:
    """
    Parse an `email.message.Message` into an `InteracRecord`, or return None
    if the message isn't a money-sent / money-received Interac notification.
    """
    subject = (msg.get("Subject") or "").strip()
    if not _is_in_scope_subject(subject):
        return None

    from_addr = parseaddr(msg.get("From", ""))[1].lower()
    body = _get_body_text(msg)

    if "interac.ca" in from_addr:
        return _parse_interac_proper(subject, body, msg)
    if "bmo.com" in from_addr:
        return _parse_bmo_notification(subject, body, msg)
    if "td.com" in from_addr or "tdcanadatrust.com" in from_addr:
        return _parse_td_notification(subject, body, msg)
    if "tangerine.ca" in from_addr:
        return _parse_tangerine_notification(subject, body, msg)

    # Sender allowlist was extended at runtime — try the generic parser.
    return _parse_generic(subject, body, msg)


# ---
# Per-format sub-parsers (thin wrappers — specialize as formats diverge)
# ---

def _parse_interac_proper(subject: str, body: str, msg: Message) -> Optional[InteracRecord]:
    return _parse_generic(subject, body, msg)


def _parse_bmo_notification(subject: str, body: str, msg: Message) -> Optional[InteracRecord]:
    return _parse_generic(subject, body, msg)


def _parse_td_notification(subject: str, body: str, msg: Message) -> Optional[InteracRecord]:
    return _parse_generic(subject, body, msg)


def _parse_tangerine_notification(subject: str, body: str, msg: Message) -> Optional[InteracRecord]:
    return _parse_generic(subject, body, msg)


def _parse_generic(subject: str, body: str, msg: Message) -> Optional[InteracRecord]:
    direction = _detect_direction(subject, body)
    if direction is None:
        return None

    amount = _parse_amount(body) or _parse_amount(subject)
    if amount is None or amount <= 0:
        return None

    from_addr = parseaddr(msg.get("From", ""))[1].lower()
    account = _parse_account(body)
    return InteracRecord(
        date=_parse_date(body, _email_received_date(msg)),
        direction=direction,
        amount=amount,
        bank=_extract_bank(from_addr, body, account),
        contact=_extract_contact(direction, subject, body),
        account=account,
        reference=_parse_reference(body),
        memo=_parse_memo(body),
        subject=subject,
    )


# ---
# Subject-level scope filter
# ---

_INTERAC_SUBJECT_RE = re.compile(
    r"interac|e[\s\-]?transfer|money\s+transfer", re.IGNORECASE
)
_OUT_OF_SCOPE_SUBJECT_RE = re.compile(
    r"\b(request(?:ed)?|declined|cancelled|canceled|expired|reminder|reject)\b",
    re.IGNORECASE,
)


def _is_in_scope_subject(subject: str) -> bool:
    if not subject:
        return False
    if not _INTERAC_SUBJECT_RE.search(subject):
        return False
    if _OUT_OF_SCOPE_SUBJECT_RE.search(subject):
        return False
    return True


# ---
# Field-level parsers
# ---

_REFERENCE_RE = re.compile(
    r"Reference\s*(?:Number)?\s*[:#]?\s*([A-Z0-9]{6,})", re.IGNORECASE
)
_MEMO_RE = re.compile(
    r"(?:Sender'?s?\s+Message|Message\s+from\s+Sender|Message)\s*:\s*"
    r"[\"\u201c]?(.+?)[\"\u201d]?\s*(?:\n|$)",
    re.IGNORECASE,
)

# Account: type words that may appear with or without a literal "Account" suffix.
# Brand prefixes (BMO, Tangerine, ...) are accepted optionally before the type.
_ACCOUNT_BANK = (
    r"(?:BMO|TD|RBC|CIBC|Scotiabank?|National\s+Bank|Tangerine|Simplii|"
    r"EQ\s+Bank|HSBC|Desjardins|Manulife|Coast\s+Capital|Vancity)"
)
_ACCOUNT_TYPE = (
    r"(?:Chequing|Checking|Savings|Joint|Personal|Business|Daily|"
    r"TFSA|RRSP|RIF|RRIF|RDSP|RESP|HISA|"
    r"Visa|Mastercard|MasterCard|Credit\s+Card|Debit\s+Card|"
    r"US\s+Dollar|Card|Account)"
)
_ACCOUNT_RE = re.compile(
    rf"(?:from|to|in|into|deposited\s+(?:in(?:to)?|to))\s+"
    rf"(?:your\s+)?"
    rf"(?P<name>(?:{_ACCOUNT_BANK}\s+)?{_ACCOUNT_TYPE}(?:\s+{_ACCOUNT_TYPE})*)"
    rf"(?:.{{0,40}}?(?:ending(?:\s+(?:in|with))?\s+|[*xX]{{2,}}\s*|[.]{{3,}}\s*)(?P<suffix>\d{{3,5}}))?",
    re.IGNORECASE | re.DOTALL,
)
# Fallback: masked digits with no preceding type word ("****1234" anywhere).
_ACCOUNT_SUFFIX_ONLY_RE = re.compile(
    r"(?:ending(?:\s+(?:in|with))?\s+|[*xX]{2,}\s*|[.]{3,}\s*)(\d{3,5})",
    re.IGNORECASE,
)

def _parse_reference(text: str) -> str:
    m = _REFERENCE_RE.search(text or "")
    return m.group(1).strip() if m else ""


def _parse_memo(text: str) -> str:
    m = _MEMO_RE.search(text or "")
    if not m:
        return ""
    memo = m.group(1).strip().strip('"').strip("\u201c\u201d").strip()
    if memo.lower().startswith(("the ", "a transfer", "your transfer")):
        return ""
    return memo


def _parse_account(text: str) -> str:
    m = _ACCOUNT_RE.search(text or "")
    if not m:
        return ""
    name = re.sub(r"\s+", " ", m.group("name").strip())
    suffix = m.group("suffix")
    return f"{name} ****{suffix}" if suffix else name


# ---
# Direction & contact extraction
# ---

# Latin character ranges supporting accented names (Émile, François, Müller).
_LATIN = r"A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u017F"
_UPPER = r"A-Z\u00C0-\u00D6\u00D8-\u00DE"
# Single name token must start uppercase (case-sensitive even when the outer
# pattern carries IGNORECASE — scoped via (?-i:...)). Inner chars allow
# letters, apostrophes, hyphens, periods (e.g. "Corp."), ampersands, digits.
_NAME_TOKEN = rf"(?-i:[{_UPPER}][{_LATIN}'\-.&0-9]*)"
_NAME_GROUP = rf"({_NAME_TOKEN}(?:[ \t]+{_NAME_TOKEN}){{0,4}})"

_RECEIVED_PATTERNS = [
    re.compile(r"sent\s+you\s+(?:money|\$)", re.IGNORECASE),
    re.compile(r"automatically\s+deposited", re.IGNORECASE),
    re.compile(r"you\s+(?:have\s+)?received\s+(?:a\s+)?(?:money\s+)?transfer", re.IGNORECASE),
    re.compile(r"deposited\s+into\s+your", re.IGNORECASE),
    re.compile(r"a\s+money\s+transfer\s+from", re.IGNORECASE),
]
_SENT_PATTERNS = [
    re.compile(r"funds\s+sent\s+to", re.IGNORECASE),
    re.compile(r"you\s+sent\s+\$", re.IGNORECASE),
    re.compile(r"your\s+transfer\s+(?:of\s+\$\S+\s+)?to\s+[^\n]{1,80}?\s+(?:has\s+been|was)\s+sent",
               re.IGNORECASE),
    re.compile(
        r"\b(?:e[\s\-]?)?transfer\s+to\s+[^\n]{1,80}?\s+"
        r"(?:for\s+\$|has\s+been\s+(?:sent|deposited)|was\s+(?:sent|deposited))",
        re.IGNORECASE,
    ),
    re.compile(r"\bsent\s+(?:successfully\s+)?to\b", re.IGNORECASE),
]

# Patterns for extracting the counterparty name. `_NAME_GROUP` is group 1.
_CONTACT_RECEIVED_PATTERNS = [
    re.compile(rf"\b(?:money\s+)?transfer\s+from\s+{_NAME_GROUP}", re.IGNORECASE),
    re.compile(rf"\bdeposit(?:ed)?\s+from\s+{_NAME_GROUP}", re.IGNORECASE),
    re.compile(rf"\b{_NAME_GROUP}\s+sent\s+you\b", re.IGNORECASE),
    re.compile(rf"\bfrom\s+{_NAME_GROUP}\s+(?:sent|has|was|is|on)\b", re.IGNORECASE),
    re.compile(rf"\bfrom\s+{_NAME_GROUP}", re.IGNORECASE),
]
_CONTACT_SENT_PATTERNS = [
    re.compile(rf"\bsent\s+(?:successfully\s+)?to\s+{_NAME_GROUP}", re.IGNORECASE),
    re.compile(
        rf"\b(?:your\s+)?(?:e[\s\-]?)?transfer\s+(?:of\s+\$\S+\s+)?to\s+{_NAME_GROUP}"
        rf"\s+(?:for\s+\$|has\s+been|was|on)",
        re.IGNORECASE,
    ),
    re.compile(rf"\bfunds\s+sent\s+to\s+{_NAME_GROUP}", re.IGNORECASE),
    re.compile(rf"\bdeposited\s+by\s+{_NAME_GROUP}", re.IGNORECASE),
    re.compile(rf"\bto\s+{_NAME_GROUP}", re.IGNORECASE),
]

_NOISE_NAMES = {
    "your", "the", "a", "an", "my", "our", "this", "that",
    "account", "chequing", "checking", "savings", "credit",
    "card", "visa", "mastercard", "tfsa", "rrsp", "ending",
    "bmo", "td", "rbc", "cibc", "scotia", "scotiabank",
    "tangerine", "simplii", "interac",
    "amount", "reference", "memo", "note", "subject",
    "deposit", "deposited", "sent", "received",
    "you", "logan", "hi", "hello", "dear",
}


def _clean_contact_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"^(?:[Tt]he|[Aa]n?|[Yy]our|[Mm]y|[Oo]ur)\s+", "", name)
    # Strip trailing punctuation but keep `.` (preserves "Acme Corp.").
    name = name.strip(" \t,;:")
    if not name:
        return ""
    tokens = name.split()
    if not tokens:
        return ""
    if all(t.lower().rstrip(".") in _NOISE_NAMES for t in tokens):
        return ""
    return name


def _detect_direction(subject: str, body: str) -> Optional[str]:
    text = f"{subject}\n{body}"
    for p in _RECEIVED_PATTERNS:
        if p.search(text):
            return "received"
    for p in _SENT_PATTERNS:
        if p.search(text):
            return "sent"
    return None


def _extract_contact(direction: str, subject: str, body: str) -> str:
    text = f"{subject}\n{body}"
    patterns = (
        _CONTACT_RECEIVED_PATTERNS if direction == "received"
        else _CONTACT_SENT_PATTERNS
    )
    for regex in patterns:
        for m in regex.finditer(text):
            cleaned = _clean_contact_name(m.group(1))
            if cleaned:
                return cleaned
    return ""


# ---
# Bank inference
# ---

# Sender domain → canonical bank label. Bank-side notifications are the most
# reliable source; Interac-proper emails (notify@payments.interac.ca) fall
# through to body-token detection.
_BANK_DOMAIN_MAP = [
    ("bmo.com",                  "BMO"),
    ("td.com",                   "TD"),
    ("tdcanadatrust.com",        "TD"),
    ("rbc.com",                  "RBC"),
    ("rbcroyalbank.com",         "RBC"),
    ("cibc.com",                 "CIBC"),
    ("scotiabank.com",           "Scotiabank"),
    ("nbc.ca",                   "National Bank"),
    ("nbcn.ca",                  "National Bank"),
    ("tangerine.ca",             "Tangerine"),
    ("simplii.com",              "Simplii"),
    ("eqbank.ca",                "EQ Bank"),
    ("hsbc.ca",                  "HSBC"),
    ("desjardins.com",           "Desjardins"),
    ("manulife.ca",              "Manulife"),
    ("manulifebank.ca",          "Manulife"),
    ("coastcapitalsavings.com",  "Coast Capital"),
    ("vancity.com",              "Vancity"),
]

_BANK_TOKEN_RE = re.compile(
    r"\b(BMO|TD\s+Canada\s+Trust|TD|RBC|Royal\s+Bank|CIBC|Scotiabank|Scotia|"
    r"National\s+Bank|Tangerine|Simplii|EQ\s+Bank|HSBC|"
    r"Desjardins|Manulife|Coast\s+Capital|Vancity)\b",
    re.IGNORECASE,
)

_BANK_NORMALIZE = {
    "bmo":              "BMO",
    "td":               "TD",
    "td canada trust":  "TD",
    "rbc":              "RBC",
    "royal bank":       "RBC",
    "cibc":             "CIBC",
    "scotiabank":       "Scotiabank",
    "scotia":           "Scotiabank",
    "national bank":    "National Bank",
    "tangerine":        "Tangerine",
    "simplii":          "Simplii",
    "eq bank":          "EQ Bank",
    "hsbc":             "HSBC",
    "desjardins":       "Desjardins",
    "manulife":         "Manulife",
    "coast capital":    "Coast Capital",
    "vancity":          "Vancity",
}


def _extract_bank(from_addr: str, body: str, account_text: str) -> str:
    """
    Best-effort bank label inference. Returns "" if no bank can be identified.

    Order of precedence:
      1. Sender domain (bank-side notifications are authoritative).
      2. Bank token in the parsed account string ("BMO Chequing ****1234").
      3. Bank token anywhere in the body.
    """
    addr = (from_addr or "").lower()
    if "@" in addr:
        domain = addr.partition("@")[2]
        for suffix, label in _BANK_DOMAIN_MAP:
            if domain == suffix or domain.endswith("." + suffix):
                return label

    for text in (account_text, body):
        if not text:
            continue
        m = _BANK_TOKEN_RE.search(text)
        if m:
            key = re.sub(r"\s+", " ", m.group(1)).strip().lower()
            return _BANK_NORMALIZE.get(key, m.group(1))
    return ""
