"""
Cache-aware streaming ASR (FastConformer / Parakeet) for low-latency Luganda.

Wraps a NeMo cache-aware streaming model (Conformer/FastConformer transducer or
CTC). The heavy lifting — chunking mel-features, preserving encoder cache across
chunks — is done exactly as in NeMo's reference
``speech_to_text_cache_aware_streaming_infer.py`` via
``CacheAwareStreamingAudioBuffer`` + ``conformer_stream_step``.

Two entry points:
  * ``stream_array(samples)`` — faithful chunk-by-chunk pass over one audio
    array; yields the interim transcript after every chunk and returns the final
    transcript. Cache-aware streaming output equals offline output, so this is
    also used per-turn by the live server.
  * EOU-capable models (e.g. parakeet_realtime_eou_120m-v1) emit ``<EOU>``/
    ``<EOB>`` inline; we surface the raw text and let ``EndOfTurnDetector`` parse.

Requires a model *trained in streaming mode* (att_context_style: chunked_limited).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class StreamChunkResult:
    text: str          # cumulative transcript for the stream so far
    is_final: bool     # True on the last chunk of the pass


class StreamingASR:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None
        self._compute_dtype = None

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self) -> None:
        try:
            import torch
            import nemo.collections.asr as nemo_asr
            from nemo.collections.asr.models import ASRModel
        except ImportError as e:  # pragma: no cover - requires NeMo
            raise RuntimeError("NeMo not installed. Run: pip install nemo_toolkit[asr]") from e

        ckpt = self._settings.streaming_asr_checkpoint
        if ckpt and str(ckpt) and __import__("pathlib").Path(ckpt).exists():
            logger.info("Loading Luganda streaming checkpoint: %s", ckpt)
            self._model = ASRModel.restore_from(str(ckpt))
        else:
            logger.info("Loading pretrained streaming model: %s", self._settings.streaming_asr_model)
            self._model = ASRModel.from_pretrained(self._settings.streaming_asr_model)

        # Select the requested lookahead for multi-lookahead models (latency knob).
        enc = self._model.encoder
        if hasattr(enc, "set_default_att_context_size"):
            try:
                enc.set_default_att_context_size(att_context_size=list(self._settings.att_context_size))
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not set att_context_size (%s); using model default.", e)

        # Low-latency greedy decoding. Hybrid models expose cur_decoder + decoder_type.
        try:
            from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
            from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig

            if self._settings.decoder_type == "ctc":
                cfg = CTCDecodingConfig()
            else:
                cfg = RNNTDecodingConfig(fused_batch_size=-1)
            if hasattr(self._model, "cur_decoder"):
                self._model.change_decoding_strategy(cfg, decoder_type=self._settings.decoder_type)
            else:
                self._model.change_decoding_strategy(cfg)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not set streaming decoding strategy (%s); using default.", e)

        self._model.eval()
        self._compute_dtype = torch.float32  # cache-aware models require float32
        logger.info("Streaming ASR ready (att_context_size=%s).", self._settings.att_context_size)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # ── Streaming inference ────────────────────────────────────────────────────

    def _stream(self, samples: np.ndarray) -> Iterator[StreamChunkResult]:
        """Yield interim transcripts chunk-by-chunk for one audio array."""
        if self._model is None:
            raise RuntimeError("Call load() before streaming")

        import torch
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )

        buffer = CacheAwareStreamingAudioBuffer(model=self._model, online_normalization=False)
        buffer.append_audio(np.asarray(samples, dtype=np.float32), stream_id=-1)

        cache_last_channel, cache_last_time, cache_last_channel_len = (
            self._model.encoder.get_initial_cache_state(batch_size=1)
        )
        previous_hypotheses = None
        pred_out_stream = None
        transcribed_texts = None

        for step_num, (chunk_audio, chunk_lengths) in enumerate(iter(buffer)):
            drop = 0 if step_num == 0 else self._model.encoder.streaming_cfg.drop_extra_pre_encoded
            with torch.inference_mode():
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                ) = self._model.conformer_stream_step(
                    processed_signal=chunk_audio.to(self._compute_dtype),
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=buffer.is_buffer_empty(),
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out_stream,
                    drop_extra_pre_encoded=drop,
                    return_transcription=True,
                )
            yield StreamChunkResult(
                text=_first_text(transcribed_texts),
                is_final=buffer.is_buffer_empty(),
            )

    def stream_array(self, samples: np.ndarray):
        """Run a full streaming pass; return (final_text, [interim_text, ...])."""
        interims: list[str] = []
        final = ""
        for res in self._stream(samples):
            interims.append(res.text)
            final = res.text
        return final, interims

    def transcribe_turn(self, samples: np.ndarray) -> str:
        """Convenience: final transcript for the current turn's audio."""
        final, _ = self.stream_array(samples)
        return final


def _first_text(transcribed_texts) -> str:
    """Extract text from NeMo's per-batch hypothesis/list return."""
    if not transcribed_texts:
        return ""
    item = transcribed_texts[0]
    return item.text if hasattr(item, "text") else str(item)
