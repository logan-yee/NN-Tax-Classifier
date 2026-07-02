"""
email_utils.py — Shared helpers for email body/payload decoding and
generic field parsing (amounts, dates).

Used by both interac_parser.py and invoice_parser.py so the two
parsers do not duplicate body-walk, HTML-strip, or amount/date regex
logic. Pure, side-effect free.
"""

import datetime
import re
from decimal import Decimal, InvalidOperation
from email.message import Message
from email.utils import parsedate_to_datetime
from typing import Optional


# ---
# Body extraction (multipart-aware, HTML-tolerant)
# ---

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RUN_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n\s*\n+")
_HTML_ENTITIES = {
    "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&#39;": "'", "&quot;": '"', "&apos;": "'",
}


def get_body_text(msg: Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return decode_payload(part)
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                return strip_html(decode_payload(part))
        return ""
    payload = decode_payload(msg)
    if msg.get_content_type() == "text/html":
        return strip_html(payload)
    return payload


def decode_payload(part: Message) -> str:
    raw = part.get_payload(decode=True)
    if raw is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace")


def strip_html(html: str) -> str:
    text = _TAG_RE.sub(" ", html)
    for entity, replacement in _HTML_ENTITIES.items():
        text = text.replace(entity, replacement)
    text = _WS_RUN_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def email_received_date(msg: Message) -> datetime.date:
    raw = msg.get("Date")
    if not raw:
        return datetime.date.today()
    try:
        return parsedate_to_datetime(raw).date()
    except (TypeError, ValueError):
        return datetime.date.today()


# ---
# Amount / date parsing
# ---

AMOUNT_RE = re.compile(
    r"\$\s*(\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2}|\d{1,3}(?:,\d{3})*|\d+)"
)

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
DATE_RE = re.compile(
    r"\b(?P<mon>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")"
    r"\.?\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)

# ISO-style date (2025-03-14) and numeric forms (03/14/2025, 14-03-2025).
_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DATE_NUMERIC_RE = re.compile(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b")


def parse_amount(text: str) -> Optional[Decimal]:
    if not text:
        return None
    m = AMOUNT_RE.search(text)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def parse_date(text: str, fallback: datetime.date) -> datetime.date:
    """Best-effort date extraction. Tries 'March 14, 2025', '2025-03-14',
    then '03/14/2025' (assumed month-first). Returns `fallback` on no match."""
    if not text:
        return fallback

    m = DATE_RE.search(text)
    if m:
        try:
            return datetime.date(
                int(m.group("year")),
                _MONTHS[m.group("mon").lower()],
                int(m.group("day")),
            )
        except (KeyError, ValueError):
            pass

    m = _DATE_ISO_RE.search(text)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    m = _DATE_NUMERIC_RE.search(text)
    if m:
        try:
            return datetime.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            pass

    return fallback
