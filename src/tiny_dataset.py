import random
from datasets import Dataset, load_dataset


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

def snli_dataset(split="train[:100%]", max_words=12):
    raw_dataset = load_dataset("snli", split=split)
    raw_dataset = raw_dataset.filter(
        lambda x: x["hypothesis"] != "" and len(x["hypothesis"].split()) <= max_words
    )
    raw_dataset = raw_dataset.rename_column("hypothesis", "text")
    raw_dataset = raw_dataset.remove_columns([c for c in raw_dataset.column_names if c != "text"])
    return raw_dataset

def wiki_dataset(split="train[:100%]", max_words=100, min_words=50):
    raw_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    raw_dataset = raw_dataset.filter(
        lambda x: min_words <= len(x["text"].split()) <= max_words
    )
    raw_dataset = raw_dataset.remove_columns([c for c in raw_dataset.column_names if c != "text"])
    return raw_dataset