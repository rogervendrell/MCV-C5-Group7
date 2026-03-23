"""Baseline image-captioning model: ResNet encoder + LSTM decoder."""
import torch
from torch import nn
from transformers import ResNetModel

from vocabulary import TEXT_MAX_LEN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class BaselineModel(nn.Module):
    """ResNet image encoder + multi-layer LSTM character-level decoder."""

    def __init__(
        self,
        vocab_size: int,
        sos_idx: int,
        freeze_encoder: bool = False,
        num_decoder_layers: int = 1,
        encoder_name: str = 'microsoft/resnet-18',
        unfreeze_last_stage: bool = False,
        hidden_size: int = 1024,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.sos_idx = sos_idx
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size

        # Encoder
        self.encoder = ResNetModel.from_pretrained(encoder_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            if unfreeze_last_stage:
                for param in self.encoder.encoder.stages[-1].parameters():
                    param.requires_grad = True

        # Decoder components
        self.enc_proj = nn.Linear(512, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_decoder_layers,
            dropout=dropout if num_decoder_layers > 1 else 0.0,
        )
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, img: torch.Tensor, ground_truth: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = img.shape[0]

        feat = self.encoder(img)
        hidden = self.enc_proj(feat.pooler_output.view(batch_size, 512)).unsqueeze(0)
        hidden = hidden.repeat(self.num_decoder_layers, 1, 1).contiguous()
        cell = torch.zeros_like(hidden)

        if False:
        # if ground_truth is not None:  # and torch.rand(1).item() < 0.5:
            input_tokens = ground_truth[:, :-1].permute(1, 0)
            embeddings = self.dropout(self.embed(input_tokens))  # (T-1, B, H)

            lstm_out, _ = self.lstm(embeddings, (hidden, cell))  # (T-1, B, H)
            lstm_out = self.dropout(lstm_out)

            res = self.proj(lstm_out).permute(1, 2, 0)
            return res
        else:
            curr_idx = torch.full((batch_size,), self.sos_idx, device=img.device)
            all_logits = []

            for _ in range(TEXT_MAX_LEN - 1):
                token_embed = self.dropout(self.embed(curr_idx)).unsqueeze(0)  # (1, B, H)
                out, (hidden, cell) = self.lstm(token_embed, (hidden, cell))
                out = self.dropout(out)
                logits = self.proj(out.squeeze(0))  # (B, Vocab)
                all_logits.append(logits.unsqueeze(2))  # (B, Vocab, 1)

                temp = 0.5
                probs = torch.softmax(logits / temp, dim=-1)
                curr_idx = torch.multinomial(probs, 1).squeeze(1)

            return torch.cat(all_logits, dim=2)
