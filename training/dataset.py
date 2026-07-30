import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from model.vision import process_image


class TextDataset(Dataset):
    def __init__(self, text_file, tokenizer_model, seq_len=128):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_model)

        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()

        self.tokens = self.sp.encode(text)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        if idx + self.seq_len + 1 > len(self.tokens):
            idx = max(0, len(self.tokens) - self.seq_len - 1)
        x = torch.tensor(self.tokens[idx : idx + self.seq_len])
        y = torch.tensor(self.tokens[idx + 1 : idx + self.seq_len + 1])
        return x, y


class MultimodalDataset(Dataset):
    """
    Dataset loader for multi-modal vision-text instruction pairs.
    Yields (x_tokens, y_tokens, image_tensor).
    """
    def __init__(self, text_file, tokenizer_model, seq_len=128, image_size=224):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_model)
        self.seq_len = seq_len
        self.image_size = image_size

        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()

        self.tokens = self.sp.encode(text)

    def __len__(self):
        return max(1, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        if idx + self.seq_len + 1 > len(self.tokens):
            idx = max(0, len(self.tokens) - self.seq_len - 1)

        x = torch.tensor(self.tokens[idx : idx + self.seq_len])
        y = torch.tensor(self.tokens[idx + 1 : idx + self.seq_len + 1])

        # Generate a standard dummy/synthetic image tensor for pretraining if no image path is bound
        image_tensor = torch.randn(3, self.image_size, self.image_size, dtype=torch.float32)

        return x, y, image_tensor
