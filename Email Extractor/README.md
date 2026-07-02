# Email Extractor — Interac transfers + invoices + online orders

Scrapes three kinds of tax-relevant emails from an IMAP mailbox in a
single pass and writes them to a combined CSV + XLSX ready to paste alongside the PDF-extractor outputs
from `../Tax Extractor/`:

1. **Interac e-Transfer notifications** (sent + received).
2. **Invoices / receipts** — PDF attachments, inline-body invoices
   (Stripe, Amazon, Uber, Apple, PayPal, Square, QuickBooks,
   FreshBooks, Wave, Shopify), and hosted-invoice links (URL recorded
   only — never fetched).
3. **Online order receipts** — anything whose subject looks like an
   order/purchase confirmation (Amazon, eBay, Etsy, Shopify storefronts,
   Best Buy, Walmart, food delivery, etc.). Routed via a subject
   heuristic, not a fixed sender list, so new merchants are picked up
   automatically.

This is **not** wired into the classifier directly — it produces files
that the user reviews and pastes manually, same human-in-the-loop
pattern as the PDF extractor.

## Security model

- Password collected with `getpass.getpass()` every run — never echoed,
  never stored on disk, never written to logs or env vars.
- IMAP over SSL on port 993 only. Plain IMAP is never imported.
- Mailbox opened read-only (`SELECT ... readonly=True`); messages
  fetched with `BODY.PEEK[]` so the `\Seen` flag is not set.
- Server-side `SEARCH` narrows by date range + (sender substring OR
  invoice subject keyword) before any body fetch — no bulk inbox
  download.
- **No outbound HTTP.** Hosted invoice URLs are recorded in the `LINK`
  column for the user to follow manually.
- **Attached PDFs are parsed in memory and discarded** — never written
  to disk. The text-extraction and OCR fallback both operate on
  `io.BytesIO`.
- The CSV + XLSX in `output/` are the only artifacts written to disk.
- `output/` is gitignored.

**Always use a provider-issued app password, not your main account
password.** Most providers also require app passwords (or OAuth) for
IMAP access today.

## Generating an app password

| Provider | Where to generate |
|---|---|
| Gmail | https://myaccount.google.com/apppasswords (requires 2FA enabled). Host: `imap.gmail.com`. |
| Outlook / Microsoft 365 | https://account.microsoft.com/security → App passwords. Host: `outlook.office365.com`. |
| Yahoo | Account Security → App passwords. Host: `imap.mail.yahoo.com`. |
| iCloud | https://appleid.apple.com → App-Specific Passwords. Host: `imap.mail.me.com`. |

Custom domains: ask your provider for the IMAP hostname. Port is always 993.

## Installing

```bash
cd "./Email Extractor"
pip install -r requirements.txt
```

### External binaries (only needed for OCR fallback on scanned PDFs)

Digitally generated invoices (Stripe, QBO, utility bills, etc.) are
parsed by `pdfplumber` alone and need no extra binaries. The OCR
fallback only kicks in for scanned / image-only PDFs.

| Binary | Why | Install (Windows) |
|---|---|---|
| Tesseract | OCR engine — turns rendered PDF page images into text. | `winget install --id UB-Mannheim.TesseractOCR` |
| Poppler | Required by `pdf2image` to render PDF pages to images. | `conda install -c conda-forge poppler`, or download Poppler-for-Windows and add `bin\` to `PATH`. |

If either binary is missing, the script does **not** crash — it just
skips the OCR fallback. Scanned-image PDFs will silently produce no
row in that case.

## Running

```bash
cd "./Email Extractor"
python email_extractor.py
```

You'll be prompted for:

1. **Email address** — full address, e.g. `you@gmail.com`.
2. **IMAP host** — defaulted from the address domain; override if needed.
3. **App password** — input is hidden.
4. **Since (YYYY-MM-DD)** — required.
5. **Until (YYYY-MM-DD)** — defaults to today; inclusive.
6. **Contact filter** — optional case-insensitive substring against the
   parsed counterparty name. Only filters Interac transfers; invoices
   are passed through.
7. **Amount filter** — optional, e.g. `>=500`, `<=20`, `=1000`, `>0`.
   Applies to Interac `AMOUNT` and invoice `TOTAL` alike.
8. **Additional sender domains** — optional, comma-separated. Bare
   domains like `rbc.com` are auto-expanded to `*@rbc.com`.

The script connects, searches, parses, filters, and writes both
`output/email_records_<since>_<until>.csv` and the matching `.xlsx`.
Existing files prompt before overwrite.

The XLSX is the same data with native typing (date cells, currency
formatting on all money columns, bold frozen header) — handy for
direct review; the CSV is the durable, diffable artifact.

## Output schema

Both the CSV and the XLSX use the same columns. Transfer rows leave
the invoice-only columns blank and vice versa. `TYPE` is `transfer`
or `invoice`.

| Column | Used by | Description |
|---|---|---|
| TYPE | all | `transfer` (Interac), `invoice`, or `order`. |
| DATE | all | Transaction / invoice / order date parsed from the message; falls back to email received date. |
| DIRECTION | transfer | `sent` or `received`. |
| AMOUNT | all | Decimal, always positive. For transfers the `DIRECTION` column carries the sign. For invoices/orders this mirrors `TOTAL` for unified sorting. |
| BANK | transfer | Canonical bank label (`BMO`, `TD`, `RBC`, `CIBC`, `Scotiabank`, `Tangerine`, ...). May be empty. |
| CONTACT | transfer | Recipient (sent) or sender (received). May be empty if name parsing fails. |
| ACCOUNT | all | Source/destination account for transfers (`Chequing Account ****1234`), or the payment card on invoices/orders (`Visa ****7086`, `Mastercard ****1234`). Blank if no card hint found. |
| REFERENCE | transfer | Interac reference / confirmation number. |
| MEMO | transfer | Sender's message, if present. |
| VENDOR | invoice, order | Canonical vendor / merchant label resolved from sender domain (Stripe, Amazon.ca, Apple, ...) or display name, falling back to the first non-trivial line of the PDF. |
| INVOICE_NO | invoice, order | Invoice / receipt number for invoices; order number for orders. |
| SUBTOTAL | invoice, order | Pre-tax amount, if found. |
| GST_HST | invoice, order | Federal sales tax (GST/HST/TPS). |
| QST_PST | invoice, order | Provincial sales tax (QST/PST/TVQ/RST). |
| TOTAL | invoice, order | Final invoice / order total. |
| CURRENCY | invoice, order | `CAD` (default), `USD`, `EUR`, or `GBP`. |
| SOURCE | all | `interac_email`, `pdf_attachment`, `inline_body`, or `hosted_link`. |
| LINK | invoice, order | Hosted invoice/receipt URL detected in the body (Stripe-hosted invoice, Square receipt, etc.). **Never fetched** — for manual follow-up. |
| SUBJECT | all | Raw email subject — for spot-checking parser output. |

## Discovery — what gets parsed

A message is fetched and tried by each parser in dispatch order
(Interac → Order → Invoice; first non-None record wins) if it matches
**any** of:

- Sender domain on the Interac allowlist (`interac_parser.ALLOWED_SENDERS`).
- Sender domain on the inline-invoice allowlist (`invoice_parser.INLINE_INVOICE_SENDERS`):
  Stripe, Amazon, Uber, Apple, PayPal, Square, QuickBooks, FreshBooks,
  Wave, Shopify.
- Subject contains any of: `invoice`, `receipt`, `bill`,
  `payment confirmation`, `payment receipt`, `tax invoice`
  (`invoice_parser.SUBJECT_KEYWORDS`).
- Subject contains any of: `order confirmation`, `your order`,
  `order #`, `order receipt`, `order summary`, `order placed`,
  `purchase confirmation`, `your purchase`, `thanks/thank you for your order`
  (`order_parser.ORDER_SUBJECT_KEYWORDS`) — these are routed to the order
  parser regardless of sender, so any merchant's order email is picked up.
- An extra sender domain you typed at the runtime prompt.

Order classification is **subject-only** — we don't maintain a list of
merchant domains. If a subject looks like an order, it's an order;
otherwise it falls through to the invoice parser. This is intentional:
new e-commerce vendors are picked up automatically.

The interactive prompt lets you add more domains per-run without
editing the file. If your bank uses a notification address outside the
default list and you'd like it on by default, add the pattern to
`interac_parser.ALLOWED_SENDERS`.

## Out-of-scope email types

The Interac parser intentionally skips:

- Money requests (`request`/`requested` in subject).
- Cancelled / declined / expired notifications.
- Reminders.

The invoice parser intentionally skips:

- Payroll / paystubs.
- Gift-card receipts.
- Shipping / delivery notifications (these often share keywords with
  receipts but aren't financial documents).
- "Received your application/submission/message" auto-replies.

These would otherwise inflate the CSV and pollute the tax workflow.

## Troubleshooting

- **Auth fails immediately**: confirm 2FA is on and you're using the
  provider's app-password page (not your main account password).
- **Zero matches**: re-run with a wider date range, or add your bank's
  notification domain to the additional-senders prompt.
- **PDF invoice produces no row**: the PDF may be a scanned image with
  no text layer. Install Tesseract + Poppler (see above) to enable the
  OCR fallback.
- **Bad parses**: open `notebooks/debug.ipynb`, save the raw email as
  a `.eml`, and feed it to `parse_interac_email` / `parse_invoice_email`
  to see what falls through. Either tighten the regex in the relevant
  parser or add a per-format sub-parser branch.

## File layout

```
Email Extractor/
├── email_extractor.py    # entry point — prompts, IMAP, orchestration, CSV+XLSX writers
├── interac_parser.py     # Interac parser, allowlist, bank inference, InteracRecord
├── invoice_parser.py     # Invoice parser — PDF + inline body + hosted-link detection, InvoiceRecord
├── order_parser.py       # Online-order parser (subject-scoped); reuses invoice_parser internals
├── email_utils.py        # shared helpers: body decoding, amount/date regexes
├── requirements.txt      # openpyxl, pdfplumber, pdf2image, pytesseract, Pillow
├── notebooks/
│   └── debug.ipynb       # parse saved .eml fixtures (no network)
└── output/               # generated CSV + XLSX (gitignored)
```
