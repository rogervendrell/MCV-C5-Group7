"""Character-level vocabulary for the VizWiz captioning baseline."""

chars = [
    '<SOS>', '<EOS>', '<PAD>',
    ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ':', ';', '=', '?', '@', '[', ']', '_', '{', '}', '<', '>', '|', '\\', '^', '~', '`',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'á', 'é', 'í', 'ó', 'ú', 'ü', 'ñ',
]

NUM_CHAR = len(chars)
idx2char = {k: v for k, v in enumerate(chars)}
char2idx = {v: k for k, v in enumerate(chars)}

TEXT_MAX_LEN = 201
