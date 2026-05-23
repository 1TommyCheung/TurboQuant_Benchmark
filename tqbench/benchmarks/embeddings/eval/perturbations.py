"""Query perturbation generators for Layer 2a.

Each perturbation function takes a query string and returns a perturbed
variant. perturb_all() composes them into a list of 5+ variants per query.
"""
from __future__ import annotations
import random
import re

# Party-name abbreviation mappings (canonical → abbrev)
PARTY_ABBREVS: dict[str, str] = {
    r"\blee\s*&\s*lee\b": "LL",
    r"\bgloria\s+james-?civetta(?:\s+&?\s*co)?\b": "GJC",
    r"\btommy\s+cheung\b": "tomi",
    r"\btracy\s+cheuk\b": "tracy",
}

# Singlish/Cantonese sentence-final particles
SINGLISH_PARTICLES = ["lah", "liao", "leh", "hor", "lor", "meh"]

# Replacement helpers for code-switching
SINGLISH_REPLACEMENTS: dict[str, str] = {
    r"\bdid we (talk about|discuss)\b": "got talk about",
    r"\bsent\b": "send",
    r"\bdid (\w+) send\b": r"got \1 send",
}

# Date fuzzing — month names and their fuzzy alternatives
MONTH_NAMES = {
    "january": "jan", "february": "feb", "march": "mar", "april": "apr",
    "may": "may", "june": "jun", "july": "jul", "august": "aug",
    "september": "sep", "october": "oct", "november": "nov", "december": "dec",
}

FUZZY_TEMPORAL_PREFIXES = ["around", "early", "mid", "late", "sometime in"]


def inject_typo(q: str, seed: int = 0) -> str:
    """Inject a single-character typo in a word longer than 3 chars."""
    rng = random.Random(seed)
    words = q.split()
    candidates = [i for i, w in enumerate(words) if len(re.sub(r"[^\w]", "", w)) > 3]
    if not candidates:
        return q
    idx = rng.choice(candidates)
    w = words[idx]
    # Find a letter position to perturb (skip leading/trailing non-letter)
    letter_positions = [i for i, c in enumerate(w) if c.isalpha()]
    if not letter_positions:
        return q
    pos = rng.choice(letter_positions)
    op = rng.choice(["substitute", "swap", "drop", "insert"])
    if op == "substitute":
        new_char = rng.choice("abcdefghijklmnopqrstuvwxyz")
        w = w[:pos] + new_char + w[pos+1:]
    elif op == "swap" and pos < len(w) - 1 and w[pos+1].isalpha():
        w = w[:pos] + w[pos+1] + w[pos] + w[pos+2:]
    elif op == "drop":
        w = w[:pos] + w[pos+1:]
    else:  # insert
        new_char = rng.choice("abcdefghijklmnopqrstuvwxyz")
        w = w[:pos] + new_char + w[pos:]
    words[idx] = w
    return " ".join(words)


def abbreviate_parties(q: str) -> str:
    """Replace canonical party names with their abbreviations."""
    out = q
    for pattern, abbrev in PARTY_ABBREVS.items():
        out = re.sub(pattern, abbrev, out, flags=re.IGNORECASE)
    return out


def fuzz_dates(q: str, seed: int = 0) -> str:
    """Replace specific dates with fuzzy temporal phrases."""
    rng = random.Random(seed)
    out = q
    # Match "5 February 2026", "February 2026", "Feb 2026"
    pattern = r"(?:\d{1,2}\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})"

    def replace(m: re.Match) -> str:
        month = m.group(1).lower()
        year = m.group(2)
        prefix = rng.choice(FUZZY_TEMPORAL_PREFIXES)
        short_month = MONTH_NAMES[month]
        # 50% chance: "around feb 2026"; 50%: "early 2026"
        if rng.random() < 0.5:
            return f"{prefix} {short_month} {year}"
        return f"{prefix} {year}"

    out = re.sub(pattern, replace, out, flags=re.IGNORECASE)
    return out


def code_switch(q: str, seed: int = 0) -> str:
    """Add Singlish particles and code-switched phrasings."""
    rng = random.Random(seed)
    out = q
    for pattern, replacement in SINGLISH_REPLACEMENTS.items():
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    # Add a sentence-final particle ~70% of the time
    if rng.random() < 0.7:
        particle = rng.choice(SINGLISH_PARTICLES)
        out = out.rstrip(" ?.") + f" {particle}"
    return out


def lowercase_proper_nouns(q: str) -> str:
    """Lower-case the whole query — mimics how a fast user types."""
    return q.lower()


def perturb_all(q: str, seed: int = 0) -> list[str]:
    """Compose 5-10 perturbed variants of the query."""
    variants: list[str] = []
    rng = random.Random(seed)

    variants.append(inject_typo(q, seed=rng.randint(0, 1_000_000)))
    variants.append(abbreviate_parties(q))
    variants.append(fuzz_dates(q, seed=rng.randint(0, 1_000_000)))
    variants.append(code_switch(q, seed=rng.randint(0, 1_000_000)))
    variants.append(lowercase_proper_nouns(q))
    # Compositions
    variants.append(lowercase_proper_nouns(abbreviate_parties(q)))
    variants.append(fuzz_dates(abbreviate_parties(q), seed=rng.randint(0, 1_000_000)))
    variants.append(inject_typo(code_switch(q, seed=rng.randint(0, 1_000_000)),
                                seed=rng.randint(0, 1_000_000)))

    # Dedupe and drop any that equal original
    out: list[str] = []
    seen: set[str] = set()
    for v in variants:
        if v == q or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out
