"""Offline unit tests for WER + CER metrics (no server / NeMo required)."""

import math

from src.metrics import (
    char_error_rate,
    corpus_cer,
    corpus_wer,
    word_error_rate,
)


def test_identical_is_zero():
    assert word_error_rate("omwana ali wano", "omwana ali wano") == 0.0


def test_single_substitution():
    # 1 of 4 reference words wrong -> 0.25
    assert word_error_rate("a b c d", "a x c d") == 0.25


def test_insertion_counts_against_reference_length():
    # one spurious inserted word over 2 ref words -> 0.5
    assert word_error_rate("a b", "a b c") == 0.5


def test_deletion():
    # one dropped word over 3 ref words
    assert math.isclose(word_error_rate("a b c", "a c"), 1 / 3, rel_tol=1e-9)


def test_case_insensitive():
    assert word_error_rate("Omwana Ali", "omwana ali") == 0.0


def test_empty_reference_empty_hyp_is_zero():
    assert word_error_rate("", "") == 0.0


def test_empty_reference_nonempty_hyp_is_one():
    assert word_error_rate("", "spurious words") == 1.0


def test_corpus_wer_aggregates_by_word_count():
    refs = ["a b c d", "a b"]
    hyps = ["a x c d", "a b"]  # 1 error across 6 words
    assert math.isclose(corpus_wer(refs, hyps), 1 / 6, rel_tol=1e-9)


def test_cer_identical_is_zero():
    assert char_error_rate("omwana", "omwana") == 0.0


def test_cer_single_char_substitution():
    # "omwana" (6 chars), one char wrong -> 1/6
    assert math.isclose(char_error_rate("omwana", "omwana".replace("a", "e", 1)), 1 / 6, rel_tol=1e-9)


def test_cer_counts_spaces():
    # ref "a b" = 3 chars incl. space; drop the space -> 1 deletion / 3
    assert math.isclose(char_error_rate("a b", "ab"), 1 / 3, rel_tol=1e-9)


def test_cer_lower_than_wer_on_near_miss():
    # one wrong letter in a 4-word sentence: WER counts a whole word, CER one char
    ref = "omwana ali wano nnyo"
    hyp = "omwana ali wanp nnyo"   # single-char error in the 3rd word
    assert char_error_rate(ref, hyp) < word_error_rate(ref, hyp)


def test_corpus_cer_aggregates_by_char_count():
    refs = ["abcd", "ab"]
    hyps = ["abxd", "ab"]  # 1 char edit across 6 ref chars
    assert math.isclose(corpus_cer(refs, hyps), 1 / 6, rel_tol=1e-9)
