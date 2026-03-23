"""Evaluation metrics: BLEU-1, BLEU-2, ROUGE-L, METEOR, sacreBLEU."""
import evaluate

import tokenizer as tok


def decode_caption(token_ids) -> str:
    """Convert a sequence of subword token IDs back to a plain string."""
    return tok.decode(token_ids)


class Metric:
    """Wrapper around HuggingFace evaluate metrics."""

    def __init__(self):
        self.bleu      = evaluate.load('bleu')
        self.rouge     = evaluate.load('rouge')
        self.meteor    = evaluate.load('meteor')
        self.sacrebleu = evaluate.load('sacrebleu')

    def __call__(self, preds: list, refs: list) -> dict:
        """
        Args:
            preds: list of predicted token-id sequences
            refs:  list of reference token-id sequences
        Returns:
            dict with keys BLEU-1, BLEU-2, ROUGE-L, METEOR, sacreBLEU
        """
        pred_texts = [decode_caption(p) for p in preds]
        ref_texts  = [decode_caption(r) for r in refs]
        refs_bleu  = [[r] for r in ref_texts]

        bleu1      = self.bleu.compute(predictions=pred_texts, references=refs_bleu, max_order=1)['bleu']
        bleu2      = self.bleu.compute(predictions=pred_texts, references=refs_bleu, max_order=2)['bleu']
        rougel     = self.rouge.compute(predictions=pred_texts, references=ref_texts)['rougeL']
        meteor     = self.meteor.compute(predictions=pred_texts, references=ref_texts)['meteor']
        sacrebleu  = self.sacrebleu.compute(predictions=pred_texts, references=refs_bleu)['score'] / 100.0

        return {'BLEU-1': bleu1, 'BLEU-2': bleu2, 'ROUGE-L': rougel, 'METEOR': meteor, 'sacreBLEU': sacrebleu}
