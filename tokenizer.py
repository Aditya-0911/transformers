from collections import Counter
import re 

class Tokenizer:

    def __init__(self):
        super().__init__()

        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] 

        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def tokenize(self, text):

        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        tokens = text.split()
        return tokens
    
    def build_vocab(self,sentences):

        counter = Counter()

        for sentence in sentences:
            tokens = self.tokenize(sentence)
            counter.update(tokens)

        idx = len(self.special_tokens)

        for word in counter:
            if word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, sentence):

        tokens = self.tokenize(sentence)
        encoded_sentence = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        return encoded_sentence
    
    def decode(self, encoded_sentence):

        decoded_sentence = [self.idx2word.get(idx, '<UNK>') for idx in encoded_sentence]
        return ' '.join(decoded_sentence)