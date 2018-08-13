import os
import pandas as pd
import numpy as np
import math
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


## refer https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/
## for feature engineering

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if str.find(big_string, substring) != -1:
            return substring
    return np.nan


def data_preprocess(df):
    df.Age = df.Age.fillna(-1)
    df.Cabin = df.Cabin.fillna('Unknown')
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    df['Deck'] = df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
    df.Embarked = df.Embarked.fillna('U')
    df = df.filter(items=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Deck', 'Embarked'])

    return df


def data_encoding(train_set, test_set):
    # Encoding
    le_sex = LabelEncoder()
    train_set.Sex = le_sex.fit_transform(train_set.Sex)
    test_set.Sex = le_sex.fit_transform(test_set.Sex)

    le_deck = LabelEncoder()
    train_set.Deck = le_deck.fit_transform(train_set.Deck)
    test_set.Deck = le_deck.fit_transform(test_set.Deck)

    le_embarked = LabelEncoder()
    train_set.Embarked = le_embarked.fit_transform(train_set.Embarked)
    test_set.Embarked = le_embarked.fit_transform(test_set.Embarked)

    return train_set, test_set


def data_process():
    train_pd= pd.read_csv('../input/train.csv')
    test_pd = pd.read_csv('../input/test.csv')
    train_X= train_pd.iloc[:, 2:]
    train_Y = train_pd.iloc[:, 1]

    ## fill missing data
    pre_train_X = data_preprocess(train_X)
    pre_test_X = data_preprocess(test_pd)

    encoded_train_X, encoded_test_X = data_encoding(pre_train_X, pre_test_X)

    ## training
    train_x = np.array(encoded_train_X)
    label_y = np.array(train_Y)

    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(1,  activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_x, label_y, epochs=2)

    ## predict
    test_x = np.array(encoded_test_X)
    predictions = model.predict(test_x)

    result = pd.DataFrame({'PassengerId': test_pd.PassengerId, 'Survived': predictions.reshape(-1)})
    result.Survived = result.Survived.map(lambda  r: 1 if r>=0.5 else 0)

    result.to_csv(path_or_buf='../input/submission.csv', index=False)


data_process()












