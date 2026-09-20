"""Optional Jev comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass

import requests


JEV_API_URL = "https://api.typesafe.ai/v1/systemone"


@dataclass(frozen=True)
class JevPrediction:
    """One binary Jev decision with an optional dark-pattern category."""

    label_name: str
    confidence: float
    category: str | None


_CATEGORY_LABELS = {
    "forced_action": "Forced Action",
    "misdirection": "Misdirection",
    "obstruction": "Obstruction",
    "scarcity": "Scarcity",
    "sneaking": "Sneaking",
    "social_proof": "Social Proof",
    "urgency": "Urgency",
}


def predict_text_with_jev(text: str, api_key: str) -> JevPrediction:
    """Ask Jev to classify text without changing the local model path."""
    normalized = " ".join(text.strip().split())
    if not normalized:
        raise ValueError("Prediction text cannot be empty")
    if not api_key.strip():
        raise ValueError("A TypeSafe API key is required")

    response = requests.post(
        JEV_API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "jev-latest",
            "state": {"text": normalized},
            "questions": {
                "classification": {
                    "type": "choice",
                    "instructions": (
                        "Does this e-commerce text use a deceptive or manipulative "
                        "dark pattern?"
                    ),
                    "criteria": {
                        "dark_pattern": (
                            "Uses pressure, deception, obstruction, sneaking, forced "
                            "action, scarcity, urgency, social proof, or misdirection."
                        ),
                        "not_dark_pattern": (
                            "Ordinary informative or promotional text without a "
                            "deceptive or manipulative tactic."
                        ),
                    },
                },
                "category": {
                    "type": "choice",
                    "instructions": (
                        "If the text is a dark pattern, which category best describes it?"
                    ),
                    "criteria": {
                        "forced_action": "Requires an unrelated action to continue.",
                        "misdirection": "Steers attention toward an unintended choice.",
                        "obstruction": "Makes a user goal unnecessarily difficult.",
                        "scarcity": "Claims limited stock or availability.",
                        "sneaking": "Adds or hides information without clear consent.",
                        "social_proof": "Uses others' activity to influence a choice.",
                        "urgency": "Creates time pressure to act quickly.",
                    },
                },
            },
        },
        timeout=15,
    )
    response.raise_for_status()
    answers = response.json()["answers"]

    classification = answers["classification"]
    is_dark_pattern = classification["choice"] == "dark_pattern"
    category = None
    if is_dark_pattern:
        category_choice = answers["category"]["choice"]
        category = _CATEGORY_LABELS.get(category_choice, category_choice)

    return JevPrediction(
        label_name="Dark Pattern" if is_dark_pattern else "Not Dark Pattern",
        confidence=float(classification["confidence"]),
        category=category,
    )
