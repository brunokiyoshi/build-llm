import re

class SimpleTokenizer: 
    def __init__(self, corpus):

        self.vocab = sorted(set(self.tokenize(corpus)))
        self.vocab.extend(["<|endoftext|>", "<|unk|>"])
        self.encoder = {token:id for id, token in enumerate(self.vocab)}
        self.decoder = {id:token for token, id in self.encoder.items()}
    
    @staticmethod
    def tokenize(text):
        return re.split(r'([,.:;?_!"()\']|--|\s)', text)

    def encode(self, text):
        tokens = self.tokenize(text)
        tokens = [token if token in self.encoder else "<|unk|>" for token in tokens]
        encoded_tokens = [self.encoder[token] for token in tokens]
        return encoded_tokens

    def decode(self, encoded_tokens):
        decoded_tokens = [self.decoder[id] for id in encoded_tokens]
        text = "".join(decoded_tokens)
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

    @property
    def vocab_size(self):
        return self.__len__()

    def __len__(self):
        return len(self.vocab)