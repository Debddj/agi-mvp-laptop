import sentencepiece as spm
import torch

class TextDataset:
    def __init__(self, text_file, tokenizer_model, seq_len=128):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_model)

        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()

        self.tokens = self.sp.encode(text)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx+self.seq_len])
        y = torch.tensor(self.tokens[idx+1:idx+self.seq_len+1])
        return x, y
