import random
from datasets import Dataset


def make_template_dataset(n=5000, seed=42):
    random.seed(seed)
    names    = ["Alice", "Bob", "Carol", "Dave", "Eve"]
    verbs    = ["likes", "hates", "eats", "reads", "finds"]
    nouns    = ["cats", "books", "pizza", "music", "coffee"]
    adverbs  = ["often", "always", "never", "sometimes", "usually"]

    sentences = [
        f"{random.choice(names)} {random.choice(adverbs)} {random.choice(verbs)} {random.choice(nouns)} ."
        for _ in range(n)
    ]
    # Pack into blocks of ~10 sentences per example
    examples = [" ".join(sentences[i:i+10]) for i in range(0, len(sentences), 10)]
    return Dataset.from_dict({"text": examples})