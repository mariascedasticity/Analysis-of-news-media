
#############
# Libraries #
#############

import numpy as np
import pandas as pd
import requests

from tqdm.notebook import tqdm

import warnings

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.feature_extraction import text 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ### Text preprocessing


def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

df = standardize_text(df, "full_text")



#Tokenizing sentences to a list of separate words
tokenizer = RegexpTokenizer(r'\w+')

df["tokens"] = df["full_text"].apply(tokenizer.tokenize)



nltk.download("stopwords")
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
    
df['tokens'] = df['tokens'].apply(lambda x: remove_stopwords(x))



words = [word for tokens in df["tokens"] for word in tokens]
vocabulary = sorted(list(set(words)))

print("%s words total, with a vocabulary size of %s" % (len(words), len(vocabulary)))



# Lemmatization
nltk.download('wordnet')
wnl = nltk.stem.WordNetLemmatizer()

def lemmatize(s):
#'''
#'''
     s = [wnl.lemmatize(word) for word in s]
     return s

df = df.assign(lemm = df.tokens.apply(lambda x: lemmatize(x)))

df = df.drop(['tokens'], axis=1)
df = df.rename(columns={"lemm": "tokens"})



df.to_csv('data.csv', index=True)


def build_df(path):
    df = pd.read_csv(path)[["id", "female", "male", "gen_label", "tokens"]]
    
    df_target = df[(df["gen_label"] == 1) | (df["gen_label"] == 2)]
    df_notarget = df[(df["gen_label"] == 3) | (df["gen_label"] == 4)]
    
    return df_target, df_notarget

def plot_roc_curve(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
def save_obj(obj, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(obj, file)
        
def load_obj(file_name):
    with open(file_name, "rb") as file:
        file = pickle.load(file)
    return file

df_target, df_notarget = build_df("result.csv")


df_model = df_target[["tokens", "gen_label"]]
df_model.loc[df_model["gen_label"] == 2, "gen_label"]  = 0



df_corpus = df_model["tokens"]
df_labels = df_model["gen_label"]

count_vectorizer = CountVectorizer(
    max_features=10000, 
    stop_words= 'english',
    min_df = 0.05,
    max_df = 0.80,
)


X_train, X_test, y_train, y_test = train_test_split(df_corpus, df_labels, test_size=0.2, random_state=42)


X_train_words = count_vectorizer.fit_transform(X_train)
X_test_words = count_vectorizer.transform(X_test)


logit = LogisticRegression(C=1, penalty='l1', 
                           solver='liblinear', n_jobs=-1, 
                           random_state=42, class_weight="balanced")



cross_val_score(logit, X_train_words, y_train, scoring="roc_auc", cv=5).mean()


plot_roc_curve(y_test, logit.fit(X_train_words, y_train).predict_proba(X_test_words)[:, 1])


# Top 200 words by magnitude of the coefficient
most_important_idx = np.argsort(np.abs(logit.coef_[0]))[-200:]

most_important_w = logit.coef_[0][most_important_idx]
most_important_word = np.array(count_vectorizer.get_feature_names())[most_important_idx]


dict(zip(most_important_word, most_important_w))


X_train_words = X_train_words.toarray()
pred_w = logit.predict_proba(X_train_words)[:, 1]


# Calculating marginal effects of words
# ME shows a one unit change in a particular word has on the predicted probability of Y,
# holding other words from the bag-of-words fixed 

marginal_effects = {}

for word_idx in tqdm(most_important_idx):
    X_train_words_i = X_train_words.copy()
    # a one unit increase
    X_train_words_i[word_idx] = X_train_words_i[word_idx] + 1  
    # Predictions of P(y | W_i + 1)
    pred_w_i = logit.predict_proba(X_train_words_i)[:, 1] # Pr(female=1)
    
    # marginal effect
    # P(y | W_i + 1) * (1 - P(y | W_i)) * beta_i
    me = (logit.coef_[0][word_idx] / pred_w.shape[0]) * np.sum(pred_w_i * (1 - pred_w))
    
    word = count_vectorizer.get_feature_names()[word_idx]
    marginal_effects[word] = me.round(3)

sorted(marginal_effects.items(), key=lambda t: t[1], reverse=True)

