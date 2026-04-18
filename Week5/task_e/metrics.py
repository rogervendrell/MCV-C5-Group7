"""Evaluation metrics: BLEU-1, BLEU-2, ROUGE-L, METEOR, sacreBLEU.

Identical to Task1/metrics.py but without the vocabulary dependency —
all predictions and references are plain strings here.
"""
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu as sacre_corpus_bleu


class Metric:
    """Compute BLEU-1/2, ROUGE-L, METEOR and sacreBLEU from string lists."""

    def __init__(self):
        self.smoother = SmoothingFunction().method1
        self.rouge    = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def __call__(self, preds: list[str], refs: list) -> dict:
        pred_texts = [p.strip() for p in preds]
        refs_multi = [self._to_reference_list(ref) for ref in refs]
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

        rouge_scores = [
            self.rouge.score(pred, ref)['rougeL'].fmeasure
            for pred, ref in zip(pred_texts, refs_single)
        ]
        rougel = sum(rouge_scores) / max(len(rouge_scores), 1)

        meteor_scores = [
            meteor_score([r.split() for r in ref_list], pred.split())
            for pred, ref_list in zip(pred_texts, refs_multi)
        ]
        meteor = sum(meteor_scores) / max(len(meteor_scores), 1)

        sacrebleu = (
            sacre_corpus_bleu(
                pred_texts,
                list(map(list, zip(*_pad_refs(refs_multi)))),
            ).score
            / 100.0
        )

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
                refs = [item.strip() for item in value if item and item.strip()]
                return refs or ['']
        return [str(value).strip()]


def _pad_refs(refs_multi: list[list[str]]) -> list[list[str]]:
    max_refs = max(len(x) for x in refs_multi)
    padded = []
    for ref_list in refs_multi:
        if len(ref_list) < max_refs:
            ref_list = ref_list + [ref_list[-1]] * (max_refs - len(ref_list))
        padded.append(ref_list)
    return padded
