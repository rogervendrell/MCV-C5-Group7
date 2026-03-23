"""Baseline image-captioning model: ResNet-18 encoder + GRU decoder."""
import torch
from torch import nn
from transformers import ResNetModel

from vocabulary import TEXT_MAX_LEN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class BaselineModel(nn.Module):
    """ResNet-18 image encoder + multi-layer GRU character-level decoder.

    The encoder produces a 512-d feature vector that seeds all GRU hidden
    states. At each decoding step the previous output is fed back as input
    (open-loop teacher-free, matching the original notebook).
    """

    def __init__(self, vocab_size: int, sos_idx: int, freeze_encoder: bool = False, num_decoder_layers: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.sos_idx = sos_idx
        self.num_decoder_layers = num_decoder_layers

        # Encoder
        self.encoder = ResNetModel.from_pretrained('microsoft/resnet-18')
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Decoder components
        self.gru   = nn.GRU(512, 512, num_layers=num_decoder_layers)
        self.embed = nn.Embedding(vocab_size, 512)
        self.proj  = nn.Linear(512, vocab_size)

    def forward(self, img: torch.Tensor, ground_truth: torch.Tensor | None = None, epoch: int = 0) -> torch.Tensor:
        """
        Args:
            img:          (B, 3, 224, 224)
            ground_truth: (B, TEXT_MAX_LEN) token indices, optional.
                          When provided, teacher forcing is used (single GRU call, no loop).
                          When None, autoregressive decoding is used.
        Returns:
            logits: (B, vocab_size, TEXT_MAX_LEN)
        """
        batch_size = img.shape[0]
        feat = self.encoder(img)
        hidden = feat.pooler_output.view(batch_size, 512).unsqueeze(0)
        hidden = hidden.repeat(self.num_decoder_layers, 1, 1).contiguous()

        # --- SCHEDULED SAMPLING ---
        # Starts at 1.0 (all teacher forcing) and drops 0.05 every epoch.
        # By epoch 20, it's 0.0 (all autoregressive).
        tf_prob = max(0.0, 1.0 - (epoch * 0.05)) 

        use_teacher_forcing = ground_truth is not None   
        if use_teacher_forcing: # and torch.rand(1).item() < 0.5:
            # Teacher forcing: single GRU call over embedded ground-truth tokens.
            # Input:  [embed(gt[0]), ..., embed(gt[T-2])]  — gt already starts with <SOS>
            # Output: T-1 GRU hidden vectors for positions 1..T-1.
            # Position 0 is always the bare <SOS> embedding (no GRU step), matching
            # the autoregressive path where outputs[0] = inp = SOS before the loop.
            input_tokens = ground_truth[:, :-1].permute(1, 0)
            embeddings = self.embed(input_tokens) # (T-1, B, 512)

            gru_out, _ = self.gru(embeddings, hidden) # (T-1, B, 512)

            res = self.proj(gru_out).permute(1, 2, 0)
            return res
        else:
            # Autoregressive: one token at a time, carrying the hidden state forward.
            curr_idx = torch.full((batch_size,), self.sos_idx, device=img.device)
            all_logits = []
            for _ in range(TEXT_MAX_LEN - 1):
                token_embed = self.embed(curr_idx).unsqueeze(0) # (1, B, 512)
                out, hidden = self.gru(token_embed, hidden)
                logits = self.proj(out.squeeze(0)) # (B, Vocab)
                all_logits.append(logits.unsqueeze(2)) # (B, Vocab, 1)

                # --- TEMPERATURE EXPERIMENT ---
                temp = 0.3 # Change this value to 0.5, 0.8, or 1.2
                probs = torch.softmax(logits / temp, dim=-1)
                curr_idx = torch.multinomial(probs, 1).squeeze(1) 
                # ------------------------------

            return torch.cat(all_logits, dim=2)
