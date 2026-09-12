from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    canary_model: str = "nvidia/canary-1b-flash"
    salm_model: str = "nvidia/salm"
    target_lang: str = "en"
    host: str = "0.0.0.0"
    port: int = 8024
    word_boost_factor: float = 15.0
    beam_size: int = 8
    hf_token: str = ""

    # Paths
    word_lists_dir: Path = Path("data/word_lists")
    ngram_lm_path: Path = Path("outputs/ngram_lm/luganda_5gram.arpa")
    spell_dict_path: Path = Path("data/spell/luganda_frequency.txt")
    finetuned_checkpoint: Path | None = None  # set after fine-tuning

    # ── Streaming / live ──────────────────────────────────────────────────────
    # Cache-aware streaming FastConformer/Parakeet checkpoint (.nemo or NGC/HF id).
    # parakeet_realtime_eou_120m-v1 emits inline <EOU>/<EOB> turn tokens.
    streaming_asr_model: str = "nvidia/parakeet_realtime_eou_120m-v1"
    streaming_asr_checkpoint: Path | None = None  # local Luganda fine-tune
    # Cache-aware lookahead: [left, right] att context (right ~ latency budget).
    att_context_size: tuple[int, int] = (70, 13)
    decoder_type: str = "rnnt"  # rnnt or ctc (hybrid models)
    # End-of-turn detection
    turn_stop_secs: float = 0.8        # trailing silence that ends a turn (VAD)
    turn_min_words: int = 1            # min words before a turn can be finalized
    turn_max_buffer_words: int = 60    # hard flush even without EOU/silence
    use_eou_tokens: bool = True        # honour inline <EOU>/<EOB> if the model emits them
    # Sortformer streaming diarizer
    diar_streaming_model: str = "nvidia/diar_streaming_sortformer_4spk-v2"
    diar_chunk_secs: float = 1.0

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
