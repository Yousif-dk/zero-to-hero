#!/usr/bin/env python3
import re
import argparse
import unicodedata

# WhatsApp timestamp patterns
TS_PATTERNS = [
    r"\[\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}[:.]\d{2}(?::\d{2})?(?:\s*[AP]M)?\]\s*",
    r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}[:.]\d{2}(?::\d{2})?(?:\s*[AP]M)?\s*-\s*"
]
TS_START = re.compile(rf"^(?:{TS_PATTERNS[0]}|{TS_PATTERNS[1]})")

EDITED_TAG = re.compile(r"(?:<\s*This message was edited\s*>)", re.IGNORECASE)
IMAGE_OMITTED = re.compile(r"^(?:image omitted|<image omitted>)\s*$", re.IGNORECASE)

# Full emoji regex (covers faces, symbols, flags, pictographs, etc.)
EMOJI_PATTERN = re.compile(
    "["                                  # start of char class
    "\U0001F600-\U0001F64F"              # emoticons
    "\U0001F300-\U0001F5FF"              # symbols & pictographs
    "\U0001F680-\U0001F6FF"              # transport & map
    "\U0001F1E0-\U0001F1FF"              # flags (iOS)
    "\U00002500-\U00002BEF"              # Chinese characters & symbols
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"                             # variation selector
    "\u3030"
    "]+", flags=re.UNICODE
)

def strip_timestamp(s: str) -> str:
    return TS_START.sub("", s, count=1)

def clean_fragment(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("[U+200E]", "").replace("\u200e", "")
    s = EDITED_TAG.sub("", s)
    s = EMOJI_PATTERN.sub("", s)  # remove all emojis
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def main():
    ap = argparse.ArgumentParser(description="Clean WhatsApp export text for Transformer training.")
    ap.add_argument("input", help="Path to WhatsApp exported .txt file")
    ap.add_argument("output", help="Path to cleaned text output file")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    cleaned_lines = []
    buffer = ""

    for raw in lines:
        if not raw.strip():
            if buffer:
                buffer += "\n"
            continue

        if TS_START.match(raw):
            if buffer:
                msg = clean_fragment(buffer)
                if msg and not IMAGE_OMITTED.match(msg):
                    cleaned_lines.append(msg)
                buffer = ""
            buffer = strip_timestamp(raw)
        else:
            if buffer:
                buffer += "\n" + raw
            else:
                buffer = raw

    if buffer:
        msg = clean_fragment(buffer)
        if msg and not IMAGE_OMITTED.match(msg):
            cleaned_lines.append(msg)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_lines) + "\n")

if __name__ == "__main__":
    main()
