# import streamlit as st
import requests
from bs4 import BeautifulSoup
import pickle
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import nltk
from gensim.models import KeyedVectors
import numpy as np
import psycopg2
import smtplib

classifier = pickle.load(open("trained_model.pkl", "rb"))
encoder = KeyedVectors.load("encoder.kv")


def fetch_html_content(url):
    response = requests.get(url)
    return response.content


def scrape_posts(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    posts = []

    for post in soup.find_all(class_='post'):
        username = post.find('strong').text.strip()[:7]
        post_text = post.find('strong').text.strip()[8:]
        posts.append((username, post_text))
        # print(username, post_text)

    return posts



def classify_post(post):
    return "Cyberbullying" if len(post.split()) > 10 else "Not Cyberbullying"


#
# nltk.download("punkt")
# nltk.download("stopwords")


def process(text):
    text = re.sub(r'[^\w\s]', " ", text)
    stopword_ = stopwords.words("english")
    clean_text = []
    stemmer = PorterStemmer()
    tokenizer = TweetTokenizer()
    for word in tokenizer.tokenize(text):
        if word not in stopword_:
            clean_text.append(stemmer.stem(word))
    return clean_text


def build_connection_with_database():
    conn = psycopg2.connect(database="cyberbul", host="localhost", port="5432", user="postgres", password="123")
    return conn


def close_connection_with_database(cur, conn):
    conn.commit()
    cur.close()
    conn.close()


map = {0: 'not_cyberbullying', 1: 'gender', 2: 'religion', 3: 'race/region', 4: 'age', 5: 'ethnicity'}

conn = build_connection_with_database()
cur = conn.cursor()
cur.execute("SELECT * FROM users")
data = cur.fetchall()
# print(data)
bullers = {}
for val in data:
    username_ = val[0]
    email_ = val[1]
    html_content = fetch_html_content(username_)
    post = scrape_posts(html_content)
    posts = []
    for i in range(len(post)):
        username = post[i][0]
        post_text = post[i][1]
        posts.append((username, post_text))
    for username, post_text in posts:
        process_post = process(post_text)
        count = 0
        vec = np.zeros(200, )
        for word in process_post:
            if word in encoder:
                vec += encoder[word]
                count += 1
        final_vec = vec / count
        prediction = classifier.predict([final_vec])
        val = map[int(prediction[0])]
        # print(prediction)
        if val == "not_cyberbullying":
            continue
        else:
            if email_ not in bullers:
                bullers[email_] = [username]
            else:
                bullers[email_].append(username)
# print(bullers)
sender_email = "shreekantpukale0@gmail.com"
password = "idoh gdte pjuo sasb"
server = smtplib.SMTP("smtp.gmail.com", 587)
server.starttls()
server.login(sender_email, password)

for email, list in bullers.items():
    message = f"These people might be Cyber bullying on you {list}"
    rec_email = email
    server.sendmail(sender_email, rec_email, message)
    # print("mail sent to ")
