from collections import Counter

"""word-level vocabulary for the VizWiz captioning baseline."""

# Special Tokens
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.idx2word = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.word2idx = {v: k for k, v in self.idx2word.items()}
        self.freq_threshold = freq_threshold

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            if not isinstance(sentence, str):
                continue
            
            # TODO: try nltk.tokenize.word_tokenize for punctuaion handling
            for word in sentence.lower().split():
                frequencies[word] += 1
            
        # Reset indices starting after special tokens
        current_idx = 4
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                if word not in self.word2idx:
                    self.word2idx[word] = current_idx
                    self.idx2word[current_idx] = word
                    current_idx += 1
    
    def __len__(self):
        return len(self.word2idx)

    def encode(self, word):
        """String -> Index (Safe)"""
        return self.word2idx.get(word, self.word2idx['<UNK>'])

    def decode(self, indices):
        """List of Indices -> List of Strings (Safe)"""
        return [self.idx2word.get(int(i), '<UNK>') for i in indices]

TEXT_MAX_LEN = 40
