import random
from datasets import Dataset


def make_template_dataset(n=5000, seed=42):
    random.seed(seed)
    names    = ["Alice", "Bob", "Carol", "Dave", "Eve"]
    verbs    = ["likes", "hates", "eats", "reads", "finds"]
    nouns    = ["cats", "books", "pizza", "music", "coffee"]
    adverbs  = ["often", "always", "never", "sometimes", "usually"]

    # One sentence per example so tokenized blocks don't straddle sentence boundaries,
    # which would cause EOS tokens to appear at position 0 of a block.
    sentences = [
        f"{random.choice(names)} {random.choice(adverbs)} {random.choice(verbs)} {random.choice(nouns)} ."
        for _ in range(n)
    ]
    return Dataset.from_dict({"text": sentences})