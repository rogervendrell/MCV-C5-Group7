"""Evaluation metrics: BLEU-1, BLEU-2, ROUGE-L, METEOR, sacreBLEU."""
import evaluate

def decode_caption(token_ids, vocab) -> str:
    """Convert a sequence of word indices back to a spaced string."""
    decoded_words = vocab.decode(token_ids)
    cleaned_words = [w for w in decoded_words if w not in ('<SOS>', '<EOS>', '<PAD>')]
    return ' '.join(cleaned_words)


class Metric:
    """Wrapper around HuggingFace evaluate metrics."""

    def __init__(self):
        self.bleu      = evaluate.load('bleu')
        self.rouge     = evaluate.load('rouge')
        self.meteor    = evaluate.load('meteor')
        self.sacrebleu = evaluate.load('sacrebleu')

    def __call__(self, preds_ids: list, refs_texts: list, vocab) -> dict:
        """
        Args:
            preds_ids:  list of predicted token-id sequences (from the model)
            refs_texts: list of lists of strings (from the dataloader/dataset)
            vocab:      the Vocabulary object used for decoding
        Returns:
            scores dict with keys BLEU-1, BLEU-2, ROUGE-L, METEOR, sacreBLEU
        """
        pred_texts = [decode_caption(p, vocab) for p in preds_ids]

        bleu1      = self.bleu.compute(predictions=pred_texts, references=refs_texts, max_order=1)['bleu']
        bleu2      = self.bleu.compute(predictions=pred_texts, references=refs_texts, max_order=2)['bleu']
        rougel     = self.rouge.compute(predictions=pred_texts, references=refs_texts)['rougeL']
        meteor     = self.meteor.compute(predictions=pred_texts, references=refs_texts)['meteor']

        return {'BLEU-1': bleu1, 'BLEU-2': bleu2, 'ROUGE-L': rougel, 'METEOR': meteor}
