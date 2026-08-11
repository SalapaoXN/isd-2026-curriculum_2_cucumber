"""
Deterministic regex pre-cleaner for raw OCR text.

Runs as step 0 of the extraction pipeline.  It only repairs UNIVERSAL
OCR artifacts so that anchor-based extraction can trust its two anchors:

    1. 8-digit course codes          (e.g. 06046400, 9064xxxx)
    2. Credit groups X(X-X-X)        (e.g. 3(3-0-6), 3(X-X-X))

Every rule below is script/structure based (digits, table borders, mixed
Latin+Thai script) and is therefore valid for ANY university's OCR output.
No rule references a specific programme, institution or watermark text.

Rules:
    - Garbage wildcard codes (XWXN / XOWX / XNWWX) -> "XXXXXXXX".
    - Restore a dropped leading '0' on a standalone 7-digit code.
    - Remove standalone / trailing table-border garbage ("|", "\\").
    - Drop short mixed Latin+Thai tokens (scan/watermark bleed-through).
    - Strip trailing Thai connectors from mostly-English lines.
    - Strip leading and trailing whitespace on every line.
"""
import re

# Garbage OCR wildcard course codes: 3-6 letters drawn from X/W/N/O
# (e.g. XWXN, XOWX, XNWWX) produced instead of the standard "XXXXXXXX".
# Requiring at least one 'X' keeps real words like "NOW" / "OWN" / "WON" intact.
_GARBAGE_WILDCARD_RE = re.compile(r"^(?=[XWNO]*X)[XWNO]{3,6}$", re.IGNORECASE)

# A 7-digit course code that lost its leading zero (e.g. 06026100 -> 6026100).
# The leading digit is dropped from a "0xxxxxxx" code, so the leftover always
# starts with 1-9; strings already starting with '0' are truncated codes, not
# dropped-zero codes, and are left untouched.  A leading '0' is restored to
# reach the universal 8-digit code anchor.
_DROPPED_LEADING_ZERO_RE = re.compile(r"(?<!\d)([1-9]\d{6})(?!\d)")

# Standalone table-border garbage lines (only pipes / backslashes).
_STANDALONE_GARBAGE_RE = re.compile(r"^[|\\]+$")

# Trailing table-border garbage (pipes / backslashes at end of line).
_TRAILING_GARBAGE_RE = re.compile(r"[|\\]+\s*$")

# Scan / watermark bleed-through: very short tokens that mix Latin and Thai
# letters (e.g. "ชSิ", "Lโม", "IIIภI") which can never be a course name,
# code or credit line.  Script-mixing, so it holds for any bilingual OCR.
_MIXED_WATERMARK_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*[\u0e00-\u0e7f])[^\s]{1,6}$")

# Trailing Thai connector words on a mostly-English line
# (e.g. "INTELLIGIENCE TECHNOLOGY หรือ" -> "INTELLIGIENCE TECHNOLOGY").
_TRAILING_THAI_ON_ENG_RE = re.compile(r"[\u0e00-\u0e7f]+\s*$")

_STANDARD_WILDCARD = "XXXXXXXX"


def pre_clean_with_regex(raw_text: str) -> str:
    """Clean raw OCR text with deterministic, universal rules.

    - Converts garbage wildcard codes (XWXN / XOWX / XNWWX ...) into "XXXXXXXX".
    - Restores a dropped leading '0' on a standalone 7-digit course code.
    - Removes standalone / trailing table-border garbage ("|", "\\").
    - Drops short mixed Latin+Thai tokens (scan/watermark bleed-through).
    - Strips trailing Thai connectors from mostly-English lines.
    - Strips leading and trailing whitespace on every line.
    """
    cleaned_lines = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if _GARBAGE_WILDCARD_RE.match(line):
            line = _STANDARD_WILDCARD
        else:
            line = _DROPPED_LEADING_ZERO_RE.sub(r"0\1", line)

        if _STANDALONE_GARBAGE_RE.match(line):
            continue

        line = _TRAILING_GARBAGE_RE.sub("", line).strip()
        if not line:
            continue

        # Drop short mixed Latin+Thai tokens (scan/watermark bleed-through).
        if _MIXED_WATERMARK_RE.match(line):
            continue

        # Strip trailing Thai connectors from mostly-English lines
        # (e.g. "INTELLIGIENCE TECHNOLOGY หรือ" -> "INTELLIGIENCE TECHNOLOGY").
        if re.search(r"[A-Za-z]", line) and re.search(r"[\u0e00-\u0e7f]", line):
            line = _TRAILING_THAI_ON_ENG_RE.sub("", line).strip()
            if not line:
                continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)
