"""Pure-function tests for the category mapping. No model load needed."""
from core.classify import categorize


def test_fresh_high_confidence():
    assert categorize("fresh", 99.4) == ("Fresh", "6 days")


def test_fresh_boundary():
    assert categorize("fresh", 85.0)[0] == "Fresh"
    assert categorize("fresh", 84.9) == ("Consume Soon", "1-2 days")


def test_rotten_always_rotten():
    assert categorize("rotten", 99.0) == ("Rotten", "Discard immediately")
    assert categorize("rotten", 50.0) == ("Rotten", "Discard immediately")
