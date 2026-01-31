import sentencepiece as spm
import os

def train_tokenizer(
    input_file="data/sample.txt",
    model_prefix="tokenizer",
    vocab_size=2000
):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0
    )

if __name__ == "__main__":
    os.makedirs("tokenizer", exist_ok=True)
    train_tokenizer()
