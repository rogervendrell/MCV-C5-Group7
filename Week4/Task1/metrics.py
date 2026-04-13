"""Evaluation metrics: BLEU-1, BLEU-2, ROUGE-L, METEOR, sacreBLEU.

Predictions are always plain strings (decoded by the GPT-2 BPE tokenizer
before reaching this module), so no custom vocabulary decoding is needed.
"""
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu as sacre_corpus_bleu


class Metric:
    """Wrapper without Hugging Face evaluate."""

    def __init__(self):
        self.smoother = SmoothingFunction().method1
        self.rouge    = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def __call__(self, preds: list[str], refs: list) -> dict:
        pred_texts = [p.strip() for p in preds]
        refs_multi = [self._to_reference_list(r) for r in refs]
        refs_single = [ref_list[0] for ref_list in refs_multi]

        pred_tokens = [p.split() for p in pred_texts]
        refs_tokens = [[r.split() for r in ref_list] for ref_list in refs_multi]

        bleu1 = corpus_bleu(
            refs_tokens, pred_tokens,
            weights=(1.0, 0.0, 0.0, 0.0),
            smoothing_function=self.smoother,
        )
        bleu2 = corpus_bleu(
            refs_tokens, pred_tokens,
            weights=(0.5, 0.5, 0.0, 0.0),
            smoothing_function=self.smoother,
        )

        rougel = sum(
            self.rouge.score(p, r)['rougeL'].fmeasure
            for p, r in zip(pred_texts, refs_single)
        ) / max(len(pred_texts), 1)

        meteor = sum(
            meteor_score([r.split() for r in ref_list], p.split())
            for p, ref_list in zip(pred_texts, refs_multi)
        ) / max(len(pred_texts), 1)

        sacrebleu = sacre_corpus_bleu(
            pred_texts, list(map(list, zip(*_pad_refs(refs_multi))))
        ).score / 100.0

        return {
            'BLEU-1':    bleu1,
            'BLEU-2':    bleu2,
            'ROUGE-L':   rougel,
            'METEOR':    meteor,
            'sacreBLEU': sacrebleu,
        }

    @staticmethod
    def _to_reference_list(value) -> list[str]:
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], str):
                refs = [v.strip() for v in value if v and v.strip()]
                return refs or ['']
        return [str(value).strip()]


def _pad_refs(refs_multi: list[list[str]]) -> list[list[str]]:
    max_refs = max(len(x) for x in refs_multi)
    return [
        r + [r[-1]] * (max_refs - len(r)) for r in refs_multi
    ]
