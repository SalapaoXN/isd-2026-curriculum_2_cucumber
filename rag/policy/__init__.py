"""Deterministic standalone QA over canonical academic policy data."""

from .answer import PolicyAnswer, PolicyFact, answer_policy_question
from .combined import CombinedAnswer, CurriculumLoad, answer_combined_question

__all__ = [
    "CombinedAnswer",
    "CurriculumLoad",
    "PolicyAnswer",
    "PolicyFact",
    "answer_combined_question",
    "answer_policy_question",
]
