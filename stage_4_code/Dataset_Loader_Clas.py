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

class Dataset_Loader_Text(dataset):

    data = None
    dataset_source_folder_path = None
    dataset_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

        self.max_vocab_size = 10000
        self.max_sequence_length = 200

    def clean_text(self, text):

        # lowercase
        text = text.lower()

        # remove punctuation/numbers
        text = re.sub(r'[^a-z\s]', '', text)

        # split into words
        words = text.split()

        return words

    def load_reviews(self, folder_path):

        texts = []
        labels = []

        for label_type in ['pos', 'neg']:

            current_folder = os.path.join(folder_path, label_type)

            for filename in os.listdir(current_folder):

                if filename.endswith('.txt'):

                    file_path = os.path.join(current_folder, filename)

                    with open(file_path, 'r', encoding='utf-8') as f:
                        review = f.read()

                    cleaned_words = self.clean_text(review)

                    texts.append(cleaned_words)

                    if label_type == 'pos':
                        labels.append(1)
                    else:
                        labels.append(0)

        return texts, labels

    def build_vocab(self, texts):

        counter = Counter()

        for review in texts:
            counter.update(review)

        most_common = counter.most_common(self.max_vocab_size - 2)

        vocab = {
            '<PAD>': 0,
            '<UNK>': 1
        }

        for idx, (word, _) in enumerate(most_common, start=2):
            vocab[word] = idx

        return vocab

    def text_to_sequence(self, words, vocab):

        sequence = []

        for word in words:

            if word in vocab:
                sequence.append(vocab[word])
            else:
                sequence.append(vocab['<UNK>'])

        return sequence

    def pad_sequence(self, sequence):

        if len(sequence) > self.max_sequence_length:
            sequence = sequence[:self.max_sequence_length]

        else:
            padding = [0] * (self.max_sequence_length - len(sequence))
            sequence = sequence + padding

        return sequence

    def load(self):

        print("loading text dataset...")

        train_folder = os.path.join(self.dataset_source_folder_path, "train")
        test_folder = os.path.join(self.dataset_source_folder_path, "test")

        # 1. Load raw reviews
        train_texts, y_train = self.load_reviews(train_folder)
        test_texts, y_test = self.load_reviews(test_folder)

        # 2. Build vocabulary ONLY from training data (IMPORTANT RULE in ML)
        vocab = self.build_vocab(train_texts)

        # 3. Convert to sequences
        X_train = []
        for review in train_texts:
            seq = self.text_to_sequence(review, vocab)
            seq = self.pad_sequence(seq)
            X_train.append(seq)

        X_test = []
        for review in test_texts:
            seq = self.text_to_sequence(review, vocab)
            seq = self.pad_sequence(seq)
            X_test.append(seq)

        # 4. Convert to numpy
        X_train = np.array(X_train, dtype=np.int64)
        X_test = np.array(X_test, dtype=np.int64)
        y_train = np.array(y_train, dtype=np.int64)
        y_test = np.array(y_test, dtype=np.int64)

        print("dataset loaded")
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)

        self.data = {
            "train": {
                "X": X_train,
                "y": y_train
            },
            "test": {
                "X": X_test,
                "y": y_test
            },
            "vocab_size": len(vocab)
        }

        return self.data