import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import gensim.downloader as api
import numpy as np

dataset = pd.read_csv("cyberbullying_tweets.csv")
dataset.dropna(inplace=True)
# print(dataset.isnull().sum())
nltk.download("punkt")
nltk.download("stopwords")

print("step 1")
def process(text):
    text = re.sub(r"[^\w\s]", " ", text)
    stopword_ = stopwords.words("english")
    clean_text = []
    stemmer = PorterStemmer()
    tokenizer = TweetTokenizer()
    for word in tokenizer.tokenize(text):
        if word not in stopword_:
            clean_text.append(stemmer.stem(word))
    return clean_text


labels = ['not_cyberbullying', 'gender', 'religion', 'other_cyberbullying', 'age', 'ethnicity']
map = {}
for d, i in zip(labels, range(0, len(labels))):
    map[d] = i
# print(map)
sentences = []
for i in range(dataset.shape[0]):
    sentences.append(process(dataset.tweet_text[i]))
# print(sentences)
# print(len(sentences))
print("step 2")
dataset.cyberbullying_type = dataset.cyberbullying_type.map(map)
dataset.tweet_text = sentences

sampled_df = pd.DataFrame(columns=['label', 'data'])
for label in dataset['cyberbullying_type'].unique():
    sampled_records = dataset[dataset['cyberbullying_type'] == label].sample(n=1000, random_state=42)
    sampled_df = pd.concat([sampled_df, sampled_records])
sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
print("step 3")
sampled_df.drop(columns={"label", "data"}, inplace=True)
encoder = api.load("glove-twitter-200")

x = []
drop = []
for i in range(0, sampled_df.shape[0]):
    l = sampled_df.tweet_text[i]
    temp = np.zeros(encoder.vector_size)
    count = 0
    for word in l:
        if word in encoder:
            temp += encoder[word]
            count += 1
    if count != 0:
        final_vec = temp / count
        x.append(final_vec)
    else:
        drop.append(i)
        print("skipping")

y = sampled_df.cyberbullying_type
y = y.drop(drop)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(accuracy_score(y_test, y_pred))

pickle.dump(classifier, open("trained_model.pkl", "wb"))
encoder.save("encoder.kv")
