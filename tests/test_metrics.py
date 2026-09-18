from src.evaluation.metrics import compute_cer, compute_wer


def test_wer_perfect_match():
    references = ["bonjour tout le monde"]
    hypotheses = ["bonjour tout le monde"]
    assert compute_wer(references, hypotheses) == 0.0


def test_wer_with_errors():
    references = ["bonjour tout le monde"]
    hypotheses = ["bonjour le monde"]
    assert compute_wer(references, hypotheses) > 0.0


def test_cer_perfect_match():
    references = ["bonjour"]
    hypotheses = ["bonjour"]
    assert compute_cer(references, hypotheses) == 0.0
