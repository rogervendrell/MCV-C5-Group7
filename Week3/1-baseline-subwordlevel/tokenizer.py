"""Subword-level (SentencePiece Unigram) tokenizer for VizWiz captions."""

import os
import json
import tempfile

import sentencepiece as spm

_HERE            = os.path.dirname(os.path.abspath(__file__))
_SP_MODEL_PREFIX = os.path.join(_HERE, 'sp_model')
_SP_MODEL_FILE   = _SP_MODEL_PREFIX + '.model'

# Vocabulary / sequence settings
VOCAB_SIZE   = 2000   # subword vocabulary size
TEXT_MAX_LEN = 64     # max tokens per sequence (including <SOS> and <EOS>)

# Special-token IDs (match SentencePiece bos/eos/unk/pad IDs below)
PAD_ID = 0   # <PAD>
SOS_ID = 1   # <SOS>  (BOS in SentencePiece)
EOS_ID = 2   # <EOS>
UNK_ID = 3   # <UNK>

# Module-level processor — populated by init()
_sp: spm.SentencePieceProcessor | None = None

# Auto-load if the model file already exists
if os.path.exists(_SP_MODEL_FILE):
    _sp = spm.SentencePieceProcessor()
    _sp.load(_SP_MODEL_FILE)


def _extract_captions(json_path: str) -> list[str]:
    """Return all valid, lowercased captions from a VizWiz annotation JSON."""
    with open(json_path) as f:
        raw = json.load(f)
    captions = []
    for ann in raw['annotations']:
        if ann.get('is_rejected') or ann.get('is_precanned'):
            continue
        caption = ann.get('caption', '').replace('\r', ' ').replace('\n', ' ').strip()
        if caption:
            captions.append(caption.lower())
    return captions


def _train(train_json: str) -> None:
    """Train a SentencePiece Unigram model on the training captions and save it."""
    captions = _extract_captions(train_json)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        for cap in captions:
            f.write(cap + '\n')
        tmp_path = f.name

    try:
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=_SP_MODEL_PREFIX,
            vocab_size=VOCAB_SIZE,
            model_type='unigram',
            pad_id=PAD_ID,
            bos_id=SOS_ID,
            eos_id=EOS_ID,
            unk_id=UNK_ID,
            pad_piece='<PAD>',
            bos_piece='<SOS>',
            eos_piece='<EOS>',
            unk_piece='<UNK>',
            character_coverage=0.9999,
        )
    finally:
        os.unlink(tmp_path)

    print(f"SentencePiece model trained and saved to {_SP_MODEL_FILE}", flush=True)


def init(train_json: str | None = None) -> None:
    """Ensure the SP model exists (train if needed) and load it."""
    global _sp
    if _sp is not None:
        return  # already loaded
    if not os.path.exists(_SP_MODEL_FILE):
        if train_json is None:
            raise FileNotFoundError(
                f"SP model not found at '{_SP_MODEL_FILE}'. "
                "Pass train_json to train it from scratch."
            )
        print("Training SentencePiece tokenizer …", flush=True)
        _train(train_json)
    _sp = spm.SentencePieceProcessor()
    _sp.load(_SP_MODEL_FILE)
    print(f"SentencePiece tokenizer loaded (vocab size={_sp.get_piece_size()})", flush=True)


def encode(text: str, max_len: int = TEXT_MAX_LEN) -> list[int]:
    """Encode *text* to a padded list of subword IDs with <SOS>/<EOS>."""
    if _sp is None:
        raise RuntimeError("Tokenizer not initialised — call tokenizer.init() first.")
    ids = [SOS_ID] + _sp.encode(text.lower()) + [EOS_ID]
    ids = ids[:max_len]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids


def decode(ids) -> str:
    """Decode a sequence of subword IDs back to plain text, stopping at <EOS>."""
    if _sp is None:
        raise RuntimeError("Tokenizer not initialised — call tokenizer.init() first.")
    token_ids = []
    for i in ids:
        i = int(i)
        if i == EOS_ID:
            break
        if i not in (SOS_ID, PAD_ID):
            token_ids.append(i)
    return _sp.decode(token_ids)
