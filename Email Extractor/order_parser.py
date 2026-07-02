"""
order_parser.py — Parse online order receipts.

Distinct from invoice_parser.py only in discovery scope (subject must
look like an order confirmation) and the resulting record TYPE.
Field-extraction internals are reused from invoice_parser since order
receipts use the same Total / Subtotal / Tax labels and the same PDF +
inline-body shapes. The point of a separate module is that the user
wants TYPE="order" rows in the output, separate from TYPE="invoice".

Same security posture as invoice_parser: PDFs parsed in memory only,
hosted links are recorded but never fetched.
"""

import datetime
import re
from dataclasses import dataclass
from decimal import Decimal
from email.message import Message
from email.utils import parseaddr
from typing import Optional

from email_utils import email_received_date, get_body_text, parse_date
from invoice_parser import (
    _extract_fields,
    _extract_hosted_link,
    _extract_payment_account,
    _extract_pdf_text,
    _find_pdf_attachments,
    _vendor_from_display_name,
    _vendor_from_domain,
    _vendor_from_pdf_text,
)


# ---
# Public constants (consumed by email_extractor.py to expand IMAP SEARCH)
# ---

# Subject substrings that flag online-order receipts. Broad on purpose —
# the user asked for "any online order", not just specific marketplaces.
ORDER_SUBJECT_KEYWORDS = [
    "order confirmation",
    "your order",
    "order placed",
    "order #",
    "order receipt",
    "order summary",
    "purchase confirmation",
    "your purchase",
    "thanks for your order",
    "thank you for your order",
]


# ---
# Record
# ---

@dataclass
class OrderRecord:
    date: datetime.date
    vendor: str
    order_no: str
    subtotal: Optional[Decimal]
    gst_hst: Optional[Decimal]
    qst_pst: Optional[Decimal]
    total: Decimal
    currency: str          # "CAD" | "USD" | "EUR" | ...
    source: str            # "pdf_attachment" | "inline_body"
    link: str              # hosted order/receipt URL if any; "" otherwise
    account: str           # payment card e.g. "Visa ****1234"; "" if not found
    subject: str

    def to_csv_row(self) -> dict:
        return {
            "DATE": self.date.isoformat(),
            "VENDOR": self.vendor,
            "INVOICE_NO": self.order_no,
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
            "INVOICE_NO": self.order_no,
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
# Subject scope
# ---

# Match "order confirmation", "your order", "order #1234", "order receipt",
# "thanks/thank you for your order", "purchase confirmation", "your purchase".
_ORDER_SUBJECT_RE = re.compile(
    r"\b(order\s+(?:confirmation|receipt|summary|placed)|"
    r"order\s*#|your\s+order|your\s+purchase|purchase\s+confirmation|"
    r"thank(?:s)?\s+(?:you\s+)?for\s+your\s+order)\b",
    re.IGNORECASE,
)
# Don't hijack shipping notices, refunds, or unrelated mail-list "orders".
_OUT_OF_SCOPE_SUBJECT_RE = re.compile(
    r"\b(payroll|paystub|gift\s*card|"
    r"shipping\s+confirmation|shipped|delivery\s+update|out\s+for\s+delivery|"
    r"return|refund\s+(?:request|issued)|cancelled|canceled)\b",
    re.IGNORECASE,
)


def _is_order_subject(subject: str) -> bool:
    if not subject:
        return False
    if _OUT_OF_SCOPE_SUBJECT_RE.search(subject):
        return False
    return bool(_ORDER_SUBJECT_RE.search(subject))


# ---
# Order-number extraction
# ---
#
# Order numbers vary more wildly than invoice numbers — Amazon uses dashed
# 17-char IDs (702-1234567-1234567), eBay uses long digit runs, Shopify
# stores use #1234. We allow letters + digits + the same continuation rule
# as invoice_parser (digit-only continuations to avoid swallowing words).

_ORDER_NO_RE = re.compile(
    r"(?:Order|Purchase|Confirmation)\s*(?:Number|No\.?|#|ID)\s*[:.]?\s*"
    r"([A-Z0-9][A-Z0-9\-_/]+(?:[\s\-]\d+){0,4})",
    re.IGNORECASE,
)


def _extract_order_no(text: str) -> str:
    m = _ORDER_NO_RE.search(text or "")
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1).strip())


# ---
# Record construction
# ---

def _build_record(
    *,
    msg: Message,
    subject: str,
    text: str,
    source: str,
    vendor_pdf_text: str = "",
    link: str = "",
) -> Optional[OrderRecord]:
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

    # Prefer an explicit Order # over the invoice-number heuristic.
    order_no = _extract_order_no(text) or fields["invoice_no"]

    return OrderRecord(
        date=parse_date(text, email_received_date(msg)),
        vendor=vendor,
        order_no=order_no,
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

def parse_order_email(msg: Message) -> Optional[OrderRecord]:
    """
    Return an `OrderRecord` if `msg` looks like an online order receipt,
    else None. Discrimination from invoice_parser is purely on subject —
    once a message is classified as an order, the same field extractors
    (Total / Subtotal / GST / QST / etc.) are applied.
    """
    subject = (msg.get("Subject") or "").strip()
    if not _is_order_subject(subject):
        return None

    body = get_body_text(msg)
    hosted_link = _extract_hosted_link(body)

    # 1. PDF attachment(s) — concatenate all PDF text so multi-PDF
    # bundles (e.g. invoice + receipt) yield a single combined record.
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

    # 2. Inline body — by far the dominant path for online orders.
    return _build_record(
        msg=msg,
        subject=subject,
        text=body,
        source="inline_body",
        link=hosted_link,
    )
