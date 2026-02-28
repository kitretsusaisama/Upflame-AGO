import os
import logging
from typing import List, Union, Dict, Any

try:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer
except ImportError:
    PreTrainedTokenizer = Any
    PreTrainedTokenizerFast = Any
    AutoTokenizer = Any

logger = logging.getLogger(__name__)

class UpFlameAGOTokenizer:
    """
    MNC-Grade Tokenizer wrapper for UpFlame-AGO.
    Automatically detects a locally trained SentencePiece model or falls back
    to a standard HuggingFace GPT-2 BPE tokenizer to guarantee resilience.
    """
    def __init__(self, model_path: str = None, fallback_name: str = "gpt2"):
        self.is_sentencepiece = False
        self.sp = None
        self.hf_tokenizer = None

        if model_path is None:
            # Resolve default path safely by checking relative to this file first, then relative to Current Working Directory.
            # This handles Google Colab and nested Git contexts robustly.
            file_relative_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "upflame_ago_tokenizer.model"))
            cwd_relative_path = os.path.abspath(os.path.join(os.getcwd(), "tokenizer", "upflame_ago_tokenizer.model"))

            if os.path.exists(file_relative_path):
                model_path = file_relative_path
            elif os.path.exists(cwd_relative_path):
                model_path = cwd_relative_path
            else:
                # Default back to whatever was provided
                model_path = file_relative_path

        # Resolve any generic ".." paths strictly to absolute paths so error logs are clean
        model_path = os.path.abspath(model_path)

        if os.path.exists(model_path):
            try:
                import sentencepiece as spm
                self.sp = spm.SentencePieceProcessor()
                self.sp.load(model_path)
                self.is_sentencepiece = True
                self._vocab_size = self.sp.vocab_size()
                self._eos_token_id = self.sp.eos_id()
                self._bos_token_id = self.sp.bos_id() if hasattr(self.sp, 'bos_id') else 1
                self._pad_token_id = self.sp.pad_id() if hasattr(self.sp, 'pad_id') else 0
                logger.info(f"Successfully loaded native SentencePiece tokenizer from {model_path}.")
            except Exception as e:
                logger.error(f"Failed to load SentencePiece model from {model_path}: {e}")
                self._load_fallback(fallback_name)
        else:
            logger.warning(f"Native tokenizer not found at {model_path}. Proceeding with fallback.")
            self._load_fallback(fallback_name)

    def _load_fallback(self, fallback_name: str):
        try:
            logger.info(f"Loading fallback tokenizer: {fallback_name}...")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(fallback_name)
            if self.hf_tokenizer.pad_token is None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            self._vocab_size = self.hf_tokenizer.vocab_size
            self._eos_token_id = self.hf_tokenizer.eos_token_id
            self._bos_token_id = self.hf_tokenizer.bos_token_id
            self._pad_token_id = self.hf_tokenizer.pad_token_id
            logger.info("Fallback tokenizer successfully loaded.")
        except Exception as e:
            logger.critical(f"Critical error loading fallback tokenizer: {e}")
            raise RuntimeError(f"Tokenizer initialization totally failed: {e}")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if self.is_sentencepiece:
            # SentencePiece intrinsically handles its own special tokens based on its training config.
            return self.sp.encode(text)
        else:
            enc = self.hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)
            if isinstance(enc, dict):
                return enc.get("input_ids", [])
            return enc

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if self.is_sentencepiece:
            try:
                # Some sentencepiece versions support skip_special_tokens in decode methods if wrapped
                return self.sp.decode(token_ids)
            except TypeError:
                return self.sp.decode(token_ids)
        else:
            return self.hf_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
