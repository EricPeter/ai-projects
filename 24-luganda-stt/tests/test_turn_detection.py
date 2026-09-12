"""Offline unit tests for end-of-turn detection (no audio / NeMo required)."""

from src.turn_detection import (
    EndOfTurnDetector,
    strip_turn_tokens,
)


def test_strip_tokens():
    clean, eou, eob = strip_turn_tokens("omwana ali wano <EOU>")
    assert clean == "omwana ali wano"
    assert eou and not eob

    clean, eou, eob = strip_turn_tokens("mm-hm <EOB>")
    assert clean == "mm-hm"
    assert eob and not eou

    clean, eou, eob = strip_turn_tokens("no tokens here")
    assert clean == "no tokens here"
    assert not eou and not eob


def test_no_end_while_speaking():
    d = EndOfTurnDetector(stop_secs=0.8)
    dec = d.update("omwana", is_speech=True, dt=0.5)
    assert not dec.end_of_turn
    assert dec.text == "omwana"


def test_silence_endpoint_after_stop_secs():
    d = EndOfTurnDetector(stop_secs=0.8, min_words=1)
    assert not d.update("omwana ali", is_speech=True, dt=0.5).end_of_turn
    assert not d.update("omwana ali", is_speech=False, dt=0.5).end_of_turn  # 0.5s silence
    dec = d.update("omwana ali", is_speech=False, dt=0.5)                    # 1.0s >= 0.8
    assert dec.end_of_turn and dec.reason == "silence"
    assert dec.text == "omwana ali"


def test_speech_resets_silence():
    d = EndOfTurnDetector(stop_secs=0.8, min_words=1)
    d.update("a b", is_speech=False, dt=0.5)
    d.update("a b c", is_speech=True, dt=0.5)   # resets silence
    dec = d.update("a b c", is_speech=False, dt=0.5)  # only 0.5s silence
    assert not dec.end_of_turn


def test_eou_token_ends_turn_immediately():
    d = EndOfTurnDetector(stop_secs=5.0)  # silence would never fire
    dec = d.update("omukwano gwange <EOU>", is_speech=True, dt=0.2)
    assert dec.end_of_turn and dec.reason == "eou"
    assert dec.text == "omukwano gwange"
    assert not dec.is_backchannel


def test_eob_token_is_backchannel():
    d = EndOfTurnDetector(stop_secs=5.0)
    dec = d.update("mm-hm <EOB>", is_speech=True, dt=0.2)
    assert dec.end_of_turn and dec.reason == "eob"
    assert dec.is_backchannel


def test_eou_ignored_when_disabled_falls_back_to_silence():
    d = EndOfTurnDetector(stop_secs=0.4, min_words=1, use_eou_tokens=False)
    dec = d.update("hello <EOU>", is_speech=True, dt=0.2)
    assert not dec.end_of_turn                # token ignored, still speaking
    dec = d.update("hello <EOU>", is_speech=False, dt=0.5)
    assert dec.end_of_turn and dec.reason == "silence"
    assert dec.text == "hello"                # token still stripped from surfaced text


def test_max_buffer_words_flush():
    d = EndOfTurnDetector(stop_secs=99.0, max_buffer_words=5)
    dec = d.update("a b c d e f", is_speech=True, dt=0.2)
    assert dec.end_of_turn and dec.reason == "max_words"


def test_no_end_on_pure_silence_without_words():
    d = EndOfTurnDetector(stop_secs=0.4, min_words=1)
    dec = d.update("", is_speech=False, dt=1.0)
    assert not dec.end_of_turn


def test_reset_clears_silence():
    d = EndOfTurnDetector(stop_secs=0.8, min_words=1)
    d.update("a b", is_speech=False, dt=0.7)
    d.reset()
    dec = d.update("a b", is_speech=False, dt=0.5)  # would be 1.2s without reset
    assert not dec.end_of_turn
