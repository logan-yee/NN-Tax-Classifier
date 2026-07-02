"""
invoice_parser.py — Parse invoice / receipt emails.

Pure, side-effect free (with one caveat: PDF + OCR libraries are
imported lazily so a missing binary doesn't crash the whole run, only
the OCR fallback path). Operates on `email.message.Message` and returns
an `InvoiceRecord` or `None`.

Three input shapes are handled, in this dispatch order:
  1. PDF attachment   — extract via pdfplumber; if no text layer, fall
                        back to Tesseract OCR.
  2. Inline body      — apply field regexes directly to the email body.
  3. Hosted-link only — if neither yields a parseable invoice but a
                        hosted invoice URL is in the body, return a
                        link-only record (so the user can open it).

Hosted invoice URLs are NEVER fetched. They're recorded in the `link`
field for the user to follow manually. This preserves the extractor's
no-outbound-HTTP posture.
"""

import datetime
import io
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from email.message import Message
from email.utils import parseaddr
from typing import Optional

from email_utils import (
    email_received_date,
    get_body_text,
    parse_date,
)


# ---
# Public constants (consumed by email_extractor.py to expand the IMAP SEARCH)
# ---

# Sender-domain allowlist for vendors that email inline invoices / receipts.
# email_extractor.py ORs these into the FROM portion of the IMAP search.
INLINE_INVOICE_SENDERS = [
    "*@stripe.com",
    "*@e.stripe.com",
    "*@amazon.ca",
    "*@amazon.com",
    "auto-confirm@amazon.ca",
    "auto-confirm@amazon.com",
    "*@uber.com",
    "*@uber.ca",
    "*@apple.com",
    "*@paypal.com",
    "*@paypal.ca",
    "*@squareup.com",
    "*@quickbooks.intuit.com",
    "*@freshbooks.com",
    "*@waveapps.com",
    "*@shopify.com",
]

# Subject substrings that trigger invoice scoping regardless of sender.
# Combined into the IMAP SEARCH with OR-of-SUBJECT.
SUBJECT_KEYWORDS = [
    "invoice",
    "receipt",
    "bill",
    "payment confirmation",
    "payment receipt",
    "your order",
    "tax invoice",
]


# ---
# Record
# ---

@dataclass
class InvoiceRecord:
    date: datetime.date
    vendor: str
    invoice_no: str
    subtotal: Optional[Decimal]
    gst_hst: Optional[Decimal]
    qst_pst: Optional[Decimal]
    total: Decimal
    currency: str          # "CAD" | "USD" | "EUR" | ...
    source: str            # "pdf_attachment" | "inline_body" | "hosted_link"
    link: str              # hosted invoice URL; "" if none detected
    account: str           # payment card e.g. "Visa ****1234"; "" if not found
    subject: str

    def to_csv_row(self) -> dict:
        return {
            "DATE": self.date.isoformat(),
            "VENDOR": self.vendor,
            "INVOICE_NO": self.invoice_no,
            "SUBTOTAL": f"{self.subtotal:.2f}" if self.subtotal is not None else "",
            "GST_HST":  f"{self.gst_hst:.2f}"  if self.gst_hst  is not None else "",
            "QST_PST":  f"{self.qst_pst:.2f}"  if self.qst_pst  is not None else "",
            "TOTAL":    f"{self.total:.2f}",
            "CURRENCY": self.currency,
            "SOURCE":   self.source,
            "LINK":     self.link,
            "ACCOUNT":  self.account,
            "SUBJECT":  self.subject,
        }

    def to_excel_row(self) -> dict:
        return {
            "DATE": self.date,
            "VENDOR": self.vendor,
            "INVOICE_NO": self.invoice_no,
            "SUBTOTAL": float(self.subtotal) if self.subtotal is not None else None,
            "GST_HST":  float(self.gst_hst)  if self.gst_hst  is not None else None,
            "QST_PST":  float(self.qst_pst)  if self.qst_pst  is not None else None,
            "TOTAL":    float(self.total),
            "CURRENCY": self.currency,
            "SOURCE":   self.source,
            "LINK":     self.link,
            "ACCOUNT":  self.account,
            "SUBJECT":  self.subject,
        }


# ---
# Subject scope filter
# ---

_INVOICE_SUBJECT_RE = re.compile(
    r"\b(invoice|receipt|bill(?:ing)?|payment\s+(?:confirmation|receipt)|"
    r"tax\s+invoice|your\s+(?:order|purchase)|amount\s+due)\b",
    re.IGNORECASE,
)
# Avoid hijacking paystubs / gift-card receipts / mailing-list "receipts".
_OUT_OF_SCOPE_SUBJECT_RE = re.compile(
    r"\b(payroll|paystub|pay\s+stub|gift\s*card|newsletter|"
    r"shipping\s+confirmation|shipped|delivery\s+update|"
    r"received\s+your\s+(?:application|submission|message))\b",
    re.IGNORECASE,
)


def _is_in_scope(subject: str, from_addr: str) -> bool:
    if _OUT_OF_SCOPE_SUBJECT_RE.search(subject or ""):
        return False
    if _INVOICE_SUBJECT_RE.search(subject or ""):
        return True
    # Inline-invoice senders are trusted even if the subject doesn't
    # explicitly say "invoice" (Stripe / Apple receipts often don't).
    domain = (from_addr or "").partition("@")[2].lower()
    if not domain:
        return False
    for pat in INLINE_INVOICE_SENDERS:
        pat_domain = pat.partition("@")[2].lower()
        if domain == pat_domain or domain.endswith("." + pat_domain):
            return True
    return False


# ---
# Vendor display-name map (sender domain → canonical vendor label).
# ---

_VENDOR_DOMAIN_MAP = [
    ("stripe.com",                "Stripe"),
    ("e.stripe.com",              "Stripe"),
    ("amazon.ca",                 "Amazon.ca"),
    ("amazon.com",                "Amazon"),
    ("uber.com",                  "Uber"),
    ("uber.ca",                   "Uber"),
    ("apple.com",                 "Apple"),
    ("paypal.com",                "PayPal"),
    ("paypal.ca",                 "PayPal"),
    ("squareup.com",              "Square"),
    ("quickbooks.intuit.com",     "QuickBooks"),
    ("intuit.com",                "Intuit"),
    ("freshbooks.com",            "FreshBooks"),
    ("waveapps.com",              "Wave"),
    ("shopify.com",               "Shopify"),
]


def _vendor_from_domain(from_addr: str) -> str:
    domain = (from_addr or "").partition("@")[2].lower()
    if not domain:
        return ""
    for suffix, label in _VENDOR_DOMAIN_MAP:
        if domain == suffix or domain.endswith("." + suffix):
            return label
    return ""


def _vendor_from_display_name(from_header: str) -> str:
    display, _addr = parseaddr(from_header or "")
    display = (display or "").strip().strip('"').strip()
    if not display or "@" in display:
        return ""
    return display


def _vendor_from_pdf_text(text: str) -> str:
    """Best-effort: first non-blank line of the PDF that isn't all digits
    and isn't a generic 'Invoice' header."""
    for raw in (text or "").splitlines():
        line = raw.strip()
        if len(line) < 3:
            continue
        if re.fullmatch(r"[\d\s\-./]+", line):
            continue
        if re.fullmatch(r"(?i)\s*(invoice|receipt|tax\s+invoice)\s*", line):
            continue
        return line[:80]
    return ""


# ---
# Field extraction
# ---
#
# Strategy: most invoice PDFs and inline emails put labels and values on
# the same line (e.g. "Total  CA$31.64"). We scan line by line, find the
# first line matching a label regex, and capture the LAST money value on
# that line. The "last money" rule handles tax lines that show their
# calculation inline, e.g. "HST - Canada  13% on CA$28.00  CA$3.64" where
# the actual tax is the trailing value.

# Money values with optional currency prefix. Accepts $, CAD/USD/EUR/GBP,
# and dollar-bloc prefixes (CA$, US$, AU$, NZ$, HK$, SG$).
_MONEY_FIND_RE = re.compile(
    r"(?:CAD|USD|EUR|GBP|CHF|AUD|NZD|HKD|SGD)?\s*"
    r"(?:CA|US|AU|NZ|HK|SG)?\$?\s*"
    r"([\d,]+\.\d{2})"
)

# Label regexes — only the label, no amount. Compiled once at module load.
_LABEL_SUBTOTAL = re.compile(r"(?i)\bSub[\s\-]?total\b")
_LABEL_GST_HST  = re.compile(r"(?i)\b(?:GST(?:/HST)?|HST|TPS)\b")
_LABEL_QST_PST  = re.compile(r"(?i)\b(?:QST|PST|TVQ|RST)\b")

# Total candidates in priority order. The strict "Total" label uses two
# guards: not preceded by a letter (excludes "Subtotal") and not followed
# by " excluding" (Stripe shows "Total excluding tax" as a separate row).
_LABEL_TOTAL_CANDIDATES = [
    re.compile(r"(?i)\bAmount\s+Paid\b"),
    re.compile(r"(?i)\bGrand\s+Total\b"),
    re.compile(r"(?i)\bAmount\s+(?:Due|Charged)\b"),
    re.compile(r"(?i)\bBalance\s+Due\b"),
    re.compile(r"(?<![A-Za-z])(?i:Total)(?!\s+excluding)\b"),
]

# Invoice number — allow a single space or hyphen between groups so we
# capture things like "9WHWYF8G 0002" or "2951 5016 8263" as one value.
# Continuation groups must be pure digits to avoid swallowing trailing
# words like "Date of issue" that follow the number on the same line.
_INVOICE_NO_RE = re.compile(
    r"(?:Invoice|Receipt|Order)\s*(?:Number|No\.?|#)\s*[:.]?\s*"
    r"([A-Z0-9][A-Z0-9\-_/]+(?:[\s\-]\d+){0,4})",
    re.IGNORECASE,
)

# Payment-card extraction. We require an explicit connector between the
# brand and the last-4 (ending/with, **, bullets, x's, hyphen, colon) so
# unrelated digits near a brand name don't false-match. Output format:
# "Visa ****1234" — same shape interac_parser uses for ACCOUNT.
_CARD_BRAND_RE = re.compile(
    r"\b(Visa|MasterCard|Master\s?Card|AmEx|American\s+Express|Discover|"
    r"Diners(?:\s+Club)?|JCB|UnionPay|Interac)\b"
    r"(?:\s*card)?"
    r"\s*(?:ending\s+(?:in\s+|with\s+)?|"
    r"\*{2,}\s*|"
    r"•{2,}\s*|"
    r"x{2,}\s*|"
    r"-\s*|"
    r":\s*)"
    r"(\d{4})\b",
    re.IGNORECASE,
)
# Generic "Card ending in 1234" / "Card ****1234" with no brand named.
_CARD_GENERIC_RE = re.compile(
    r"\b(?:card|account)\s*"
    r"(?:ending\s+(?:in\s+|with\s+)?|\*{2,}\s*|•{2,}\s*|x{2,}\s*)"
    r"(\d{4})\b",
    re.IGNORECASE,
)

_BRAND_NORMALIZE = {
    "visa": "Visa",
    "mastercard": "Mastercard",
    "master card": "Mastercard",
    "amex": "Amex",
    "american express": "Amex",
    "discover": "Discover",
    "diners": "Diners",
    "diners club": "Diners",
    "jcb": "JCB",
    "unionpay": "UnionPay",
    "interac": "Interac",
}


def _extract_payment_account(text: str) -> str:
    """Return a normalized payment-account label like 'Visa ****1234',
    or '' if no card hint is found. Branded matches win over generic
    'Card ending in ...' since they carry more information."""
    if not text:
        return ""
    m = _CARD_BRAND_RE.search(text)
    if m:
        brand_raw = re.sub(r"\s+", " ", m.group(1).strip().lower())
        brand = _BRAND_NORMALIZE.get(brand_raw, m.group(1).strip().title())
        return f"{brand} ****{m.group(2)}"
    m = _CARD_GENERIC_RE.search(text)
    if m:
        return f"Card ****{m.group(1)}"
    return ""


_CURRENCY_USD_RE = re.compile(r"\b(USD|US\$|U\.S\.\s*Dollars?)\b", re.IGNORECASE)
_CURRENCY_EUR_RE = re.compile(r"\b(EUR|€)\b")
_CURRENCY_GBP_RE = re.compile(r"\b(GBP|£)\b")
_CURRENCY_CAD_RE = re.compile(r"\b(CAD|CA\$|C\$|Canadian\s+Dollars?)\b", re.IGNORECASE)


def _last_money_on_label_line(text: str, label_re: re.Pattern) -> Optional[Decimal]:
    """Scan `text` line by line. On the first line whose `label_re` matches,
    return the LAST money value on that line as a Decimal. None if no line
    matches or no money is on the matched line."""
    for line in (text or "").splitlines():
        if not label_re.search(line):
            continue
        amounts = _MONEY_FIND_RE.findall(line)
        if not amounts:
            continue
        try:
            return Decimal(amounts[-1].replace(",", ""))
        except InvalidOperation:
            continue
    return None


def _extract_total(text: str) -> Optional[Decimal]:
    for label in _LABEL_TOTAL_CANDIDATES:
        value = _last_money_on_label_line(text, label)
        if value is not None and value > 0:
            return value
    return None


def _detect_currency(text: str) -> str:
    if not text:
        return "CAD"
    # Canadian prefixes / labels first — Stripe receipts say "CA$" and the
    # user is Canadian, so this is the common case.
    if _CURRENCY_CAD_RE.search(text):
        return "CAD"
    if _CURRENCY_USD_RE.search(text):
        return "USD"
    if _CURRENCY_EUR_RE.search(text):
        return "EUR"
    if _CURRENCY_GBP_RE.search(text):
        return "GBP"
    return "CAD"


def _extract_invoice_no(text: str) -> str:
    m = _INVOICE_NO_RE.search(text or "")
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1).strip())


# ---
# Hosted invoice URL detection
# ---

_HOSTED_INVOICE_URL_RE = re.compile(
    r"https?://(?:"
    r"invoice\.stripe\.com/[^\s\"'>)<]+|"
    r"pay\.stripe\.com/[^\s\"'>)<]+|"
    r"squareup\.com/receipt/[^\s\"'>)<]+|"
    r"(?:[\w.-]+\.)?intuit\.com/[^\s\"'>)<]*invoice[^\s\"'>)<]*|"
    r"(?:[\w.-]+\.)?freshbooks\.com/[^\s\"'>)<]*invoice[^\s\"'>)<]*|"
    r"[^\s\"'>)<]+\.pdf"
    r")",
    re.IGNORECASE,
)


def _extract_hosted_link(text: str) -> str:
    m = _HOSTED_INVOICE_URL_RE.search(text or "")
    return m.group(0).rstrip(".,;)") if m else ""


# ---
# PDF attachment extraction
# ---

def _find_pdf_attachments(msg: Message) -> list[bytes]:
    """Return raw PDF bytes for each application/pdf part on the message."""
    out: list[bytes] = []
    if not msg.is_multipart():
        return out
    for part in msg.walk():
        ctype = (part.get_content_type() or "").lower()
        filename = (part.get_filename() or "").lower()
        is_pdf = ctype == "application/pdf" or filename.endswith(".pdf")
        if not is_pdf:
            continue
        payload = part.get_payload(decode=True)
        if isinstance(payload, bytes) and len(payload) > 0:
            out.append(payload)
    return out


def _pdf_text_via_pdfplumber(pdf_bytes: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        return ""
    chunks: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                chunks.append(txt)
                # Tables — append flat rows so labels and values land on the
                # same line for the field regexes.
                try:
                    for table in (page.extract_tables() or []):
                        for row in table:
                            cells = [(c or "").strip() for c in row if c]
                            if cells:
                                chunks.append(" | ".join(cells))
                except Exception:
                    pass
    except Exception:
        return ""
    return "\n".join(chunks).strip()


def _pdf_text_via_ocr(pdf_bytes: bytes) -> str:
    """OCR fallback for scanned PDFs. Returns "" if Tesseract/Poppler are
    missing — caller handles the empty result by returning None."""
    try:
        from pdf2image import convert_from_bytes
    except ImportError:
        return ""
    try:
        import pytesseract
    except ImportError:
        return ""

    try:
        images = convert_from_bytes(pdf_bytes, dpi=200)
    except Exception:
        # Likely Poppler missing or PDF corrupted.
        return ""

    out: list[str] = []
    for img in images:
        try:
            out.append(pytesseract.image_to_string(img))
        except Exception:
            # Tesseract binary missing or page failed; skip.
            continue
    return "\n".join(out).strip()


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Try pdfplumber first; if the text layer is empty/tiny, fall back
    to OCR. Returns "" if both paths fail (caller will skip the record).
    NUL bytes (\\x00) are normalized to spaces — Stripe-generated PDFs use
    them as thin separators inside numbers and table cells, which breaks
    whitespace-aware regexes downstream."""
    text = _pdf_text_via_pdfplumber(pdf_bytes)
    if len(text) >= 40:
        return text.replace("\x00", " ")
    ocr = _pdf_text_via_ocr(pdf_bytes)
    chosen = ocr if len(ocr) > len(text) else text
    return chosen.replace("\x00", " ")


# ---
# Field extraction (shared across PDF + inline paths)
# ---

def _extract_fields(text: str) -> Optional[dict]:
    """Pull total/subtotal/tax/invoice_no/currency from `text`. Returns
    None if no positive total is found — the caller treats that as
    'not an invoice'."""
    total = _extract_total(text)
    if total is None or total <= 0:
        return None
    return {
        "total":      total,
        "subtotal":   _last_money_on_label_line(text, _LABEL_SUBTOTAL),
        "gst_hst":    _last_money_on_label_line(text, _LABEL_GST_HST),
        "qst_pst":    _last_money_on_label_line(text, _LABEL_QST_PST),
        "invoice_no": _extract_invoice_no(text),
        "currency":   _detect_currency(text),
    }


def _build_record(
    *,
    msg: Message,
    subject: str,
    text: str,
    source: str,
    vendor_pdf_text: str = "",
    link: str = "",
) -> Optional[InvoiceRecord]:
    fields = _extract_fields(text)
    if fields is None:
        return None

    from_header = msg.get("From", "")
    from_addr = parseaddr(from_header)[1].lower()
    vendor = (
        _vendor_from_domain(from_addr)
        or _vendor_from_display_name(from_header)
        or _vendor_from_pdf_text(vendor_pdf_text)
        or (from_addr.partition("@")[2].split(".")[0].title() if from_addr else "")
    )

    return InvoiceRecord(
        date=parse_date(text, email_received_date(msg)),
        vendor=vendor,
        invoice_no=fields["invoice_no"],
        subtotal=fields["subtotal"],
        gst_hst=fields["gst_hst"],
        qst_pst=fields["qst_pst"],
        total=fields["total"],
        currency=fields["currency"],
        source=source,
        link=link,
        account=_extract_payment_account(text),
        subject=subject,
    )


# ---
# Dispatcher
# ---

def parse_invoice_email(msg: Message) -> Optional[InvoiceRecord]:
    """
    Parse an `email.message.Message` into an `InvoiceRecord`, or None if
    the message isn't an invoice / receipt.
    """
    subject = (msg.get("Subject") or "").strip()
    from_addr = parseaddr(msg.get("From", ""))[1].lower()
    if not _is_in_scope(subject, from_addr):
        return None

    body = get_body_text(msg)
    hosted_link = _extract_hosted_link(body)

    # 1. PDF attachment(s). Stripe-style mailers ship the invoice and the
    # receipt as two PDFs in the same message — one carries the totals,
    # the other carries the card line. Concatenating all attachment text
    # before field extraction gives us the union of both.
    pdf_texts: list[str] = []
    for pdf_bytes in _find_pdf_attachments(msg):
        pdf_text = _extract_pdf_text(pdf_bytes)
        if pdf_text:
            pdf_texts.append(pdf_text)
    if pdf_texts:
        combined = "\n".join(pdf_texts)
        rec = _build_record(
            msg=msg,
            subject=subject,
            text=combined,
            source="pdf_attachment",
            vendor_pdf_text=combined,
            link=hosted_link,
        )
        if rec is not None:
            return rec

    # 2. Inline body.
    rec = _build_record(
        msg=msg,
        subject=subject,
        text=body,
        source="inline_body",
        link=hosted_link,
    )
    if rec is not None:
        return rec

    # 3. Hosted-link-only fallback. No amount parsed, but the email looks
    # like an invoice and contains a hosted invoice URL — surface a row so
    # the user can open it manually.
    if hosted_link:
        from_header = msg.get("From", "")
        vendor = (
            _vendor_from_domain(from_addr)
            or _vendor_from_display_name(from_header)
            or (from_addr.partition("@")[2].split(".")[0].title() if from_addr else "")
        )
        return InvoiceRecord(
            date=email_received_date(msg),
            vendor=vendor,
            invoice_no="",
            subtotal=None,
            gst_hst=None,
            qst_pst=None,
            total=Decimal("0.00"),
            currency="CAD",
            source="hosted_link",
            link=hosted_link,
            account=_extract_payment_account(body),
            subject=subject,
        )

    return None
