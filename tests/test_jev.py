import pytest

import src.jev as jev


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "answers": {
                "classification": {
                    "choice": "dark_pattern",
                    "confidence": 0.93,
                },
                "category": {"choice": "scarcity"},
            }
        }


def test_predict_text_with_jev_returns_comparison_prediction(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        assert url == jev.JEV_API_URL
        assert headers == {"Authorization": "Bearer test-key"}
        assert json["state"] == {"text": "Only 2 left. Buy now."}
        assert set(json["questions"]) == {"classification", "category"}
        assert timeout == 15
        return FakeResponse()

    monkeypatch.setattr(jev.requests, "post", fake_post)

    prediction = jev.predict_text_with_jev("  Only 2 left.  Buy now.  ", "test-key")

    assert prediction.label_name == "Dark Pattern"
    assert prediction.confidence == pytest.approx(0.93)
    assert prediction.category == "Scarcity"


@pytest.mark.parametrize("text,api_key", [("   ", "test-key"), ("Text", "   ")])
def test_predict_text_with_jev_rejects_missing_input(text, api_key):
    with pytest.raises(ValueError):
        jev.predict_text_with_jev(text, api_key)
