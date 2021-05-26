import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from keras import layers, Sequential
from gensim.models import KeyedVectors
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


def parse_test(file_name, pair):
    if pair:
        data = pd.read_csv(file_name, delimiter='\t', names=['word1', 'word2', 'target'])
        return data[['word1', 'word2']], data['target']
    else:
        data = pd.read_csv(file_name, delimiter='\t', names=['word', 'target'])
        return data['word'], data['target']


def encode_labels(y):
    encoder = LabelEncoder()
    encoder.fit(y)
    enc_y = encoder.transform(y)
    dum_y = np_utils.to_categorical(enc_y)
    return dum_y


def import_embeds(file_name):
    embedder = KeyedVectors.load(file_name)
    return embedder


def create_embeddings(X, embedder, pair):
    def _get_single_vector(word, embedder):
        try:
            v = embedder.get_vector(word)
        except KeyError:
            v = np.random.rand((embedder.vector_size))
        return v
    if pair:
        words = zip(list(X['word1']), list(X['word2']))
        return np.array([np.hstack((_get_single_vector(word[0], embedder).reshape(1, -1),
                                    _get_single_vector(word[1], embedder).reshape(1, -1))).reshape(600)for word in words])
    else:
        words = list(X)
        return np.array([_get_single_vector(word, embedder) for word in words])


def build_model(embed_size):
    classifier = Sequential()
    classifier.add(layers.Input(shape=(embed_size,)))
    classifier.add(layers.Dense(300, activation='relu'))
    classifier.add(layers.Dropout(0.5))
    classifier.add(layers.Dense(3, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier


def train_model(builder, embed_size, X, y):
    estimator = KerasClassifier(build_fn=builder, embed_size=embed_size, epochs=20, batch_size=10, verbose=2)
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X, y, cv=kfold)
    return results


def main():
    data_file = 'e_t_full.csv'
    pair = False
    emb_file = r'models\en\contentw\enwikiw_c_weights.kv'
    X_w, y_cat = parse_test(data_file, pair)
    y = encode_labels(y_cat)
    embedder = import_embeds(emb_file)
    X = create_embeddings(X_w, embedder, pair)
    train_model(build_model, embed_size=embedder.vector_size * (pair + 1), X=X, y=y)


if __name__ == '__main__':
    main()
