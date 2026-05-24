'''
Concrete IO class for text datasets (IMDb sentiment)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
import re
import numpy as np
from collections import Counter
from code.base_class.dataset import dataset


class Dataset_Loader_Joke(dataset):

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

        self.max_vocab_size = 10000
        self.max_seq_len = 30


# Clean the text
    def clean(self, text):

        text = text.lower()

        text = re.sub(r'[^a-z\s]', ' ', text)

        words = text.split()

        # keep only alphabetic words
        cleaned = []

        for w in words:

            if w.isalpha():

                # remove very weird words
                if len(w) > 1 and len(w) < 15:
                    cleaned.append(w)

        return cleaned

# Load the data
    def load_jokes(self, folder):

        texts = []

        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)

            with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                words = self.clean(f.read())

                texts.append(words)

        return texts

# Build the vocab
    def build_vocab(self, texts):

        counter = Counter()

        # count all words
        for t in texts:
            counter.update(t)

        vocab = {
            '<PAD>': 0,
            '<UNK>': 1
        }

        # keep only words appearing at least 2 times
        filtered_words = []

        for w, c in counter.items():

            if c >= 2:
                filtered_words.append((w, c))

        # sort by frequency
        filtered_words = sorted(
            filtered_words,
            key=lambda x: x[1],
            reverse=True
        )

        # build vocab
        for i, (w, _) in enumerate(
                filtered_words[:self.max_vocab_size - 2],
                start=2):
            vocab[w] = i

        return vocab

# Create sliding windows (input word - target word pair)
    def create_pairs(self, texts, vocab):

        X, y = [], []

        for text in texts:
            seq = [vocab.get(w, 1) for w in text]

            for i in range(1, len(seq)):
                X.append(seq[:i])
                y.append(seq[i])

        return X, y

# Pad
    def pad(self, seq):
        if len(seq) < self.max_seq_len:
            seq = seq + [0] * (self.max_seq_len - len(seq))
        else:
            seq = seq[:self.max_seq_len]
        return seq

# Load
    def load(self):

        print("loading joke dataset...")

        folder = self.dataset_source_folder_path

        texts = self.load_jokes(folder)
        vocab = self.build_vocab(texts)

        X, y = self.create_pairs(texts, vocab)

        X = np.array([self.pad(x) for x in X], dtype=np.int64)
        y = np.array(y, dtype=np.int64)

        print("X shape:", X.shape)
        print("y shape:", y.shape)

        return {
            "X_train": X,
            "y_train": y,
            "vocab_size": len(vocab),
            "vocab": vocab
        }
