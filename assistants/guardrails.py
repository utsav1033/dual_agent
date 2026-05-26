from typing import Tuple

JAILBREAK_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "you are now dan",
    "do anything now",
    "pretend you have no restrictions",
    "act as if you have no",
    "disregard your training",
    "you are a different ai",
    "jailbreak mode",
    "developer mode enabled",
    "bypass your safety",
    "override your guidelines",
]

HARMFUL_KEYWORDS = [
    "how to make a bomb",
    "how to build a bomb",
    "synthesize meth",
    "make methamphetamine",
    "child pornography",
    "child sexual abuse",
    "csam",
    "how to stalk",
    "how to poison",
    "how to murder",
    "make explosive",
    "build a weapon",
]

HARMFUL_OUTPUT_PATTERNS = [
    "here's how to make a bomb",
    "step 1: obtain the chemicals",
    "instructions for synthesizing",
    "to create the explosive",
]

REFUSAL_PHRASES = [
    "i can't", "i cannot", "i'm unable", "i won't", "i will not",
    "not able to", "against my", "harmful", "inappropriate",
    "i don't", "blocked", "filtered", "i apologize but",
    "as an ai", "i'm sorry, but i", "i'm not able to",
    "i must decline", "i'm afraid i can't",
]


def check_input_safety(text: str) -> Tuple[bool, str]:
    """Returns (is_safe, reason). False = unsafe."""
    lower = text.lower()

    for pattern in JAILBREAK_PATTERNS:
        if pattern in lower:
            return False, f"Jailbreak attempt detected."

    for kw in HARMFUL_KEYWORDS:
        if kw in lower:
            return False, f"Harmful content request detected."

    return True, "safe"


def check_output_safety(text: str) -> Tuple[bool, str]:
    """Returns (is_safe, reason). False = unsafe output."""
    lower = text.lower()
    for pattern in HARMFUL_OUTPUT_PATTERNS:
        if pattern in lower:
            return False, "Response contains potentially harmful content."
    return True, "safe"


def is_refusal(text: str) -> bool:
    """Check if a model response is a refusal."""
    lower = text.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)
