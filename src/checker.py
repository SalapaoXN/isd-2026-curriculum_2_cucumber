import re
from spellchecker import SpellChecker


class OCRSpellChecker:
    # English domain words that must never be "corrected" (programs, abbreviations, etc.)
    DOMAIN_WORDS = [
        "dsba", "ait", "bit", "coop", "gened", "genedx", "math", "prereq",
        "elec", "comp", "huma", "sci", "proj", "thesis", "intern", "excellent",
        "data", "soft", "stat", "econ", "fina", "mgmt", "analy", "engi",
    ]

    # English words to strongly bias pyspellchecker toward (kept as valid / fixed
    # to the desired spelling) by giving them a very high frequency.
    EN_BIAS_WORDS = {
        "analytics": 10_000_000,
        "project": 10_000_000,
        "bayesian": 10_000_000,
        "acquisition": 10_000_000,
        "cooperative": 10_000_000,
        "engineering": 10_000_000,
        "probability": 10_000_000,
    }

    def __init__(self, min_en_length: int = 4):
        print("Initializing English Spell Checker...")
        self.en_checker = SpellChecker()
        self.min_en_length = min_en_length
        for word in self.DOMAIN_WORDS:
            self.en_checker.word_frequency.add(word)
        for word, value in self.EN_BIAS_WORDS.items():
            self.en_checker.word_frequency.add(word, value)

    def _is_english(self, word: str) -> bool:
        """Check if word consists of English letters."""
        return bool(re.fullmatch(r"[a-zA-Z]+", word))

    def _match_casing(self, original: str, suggestion: str) -> str:
        """Re-apply original capitalization (UPPERCASE, Titlecase, or lowercase)."""
        if original.isupper():
            return suggestion.upper()
        if original.istitle():
            return suggestion.capitalize()
        return suggestion

    def correct_line(self, line: str) -> tuple[str, list[dict]]:
        """Correct English typos in a single line while preserving punctuation and numbers."""
        tokens = line.split(" ")
        corrected_tokens = []
        typos = []

        for token in tokens:
            # Separate surrounding punctuation from core word
            match = re.match(r"^([^\w]*)([a-zA-Z]+)([^\w]*)$", token)
            if not match:
                corrected_tokens.append(token)
                continue

            prefix, core_word, suffix = match.groups()

            if len(core_word) < self.min_en_length:
                corrected_tokens.append(token)
                continue

            word_lower = core_word.lower()
            if word_lower not in self.en_checker:
                suggestion = self.en_checker.correction(word_lower)
                if suggestion and suggestion != word_lower:
                    fixed_word = self._match_casing(core_word, suggestion)
                    typos.append({"original": core_word, "corrected": fixed_word})
                    corrected_tokens.append(f"{prefix}{fixed_word}{suffix}")
                    continue

            corrected_tokens.append(token)

        corrected_line = " ".join(corrected_tokens)
        return corrected_line, typos

    def process_lines(self, lines: list[str]) -> tuple[list[str], list[dict]]:
        """Process multiple lines and collect corrected lines along with a typo log."""
        corrected_lines = []
        all_typos = []

        for line_no, line in enumerate(lines, start=1):
            corrected_line, typos = self.correct_line(line)
            corrected_lines.append(corrected_line)
            for t in typos:
                t["line"] = line_no
                all_typos.append(t)

        return corrected_lines, all_typos
