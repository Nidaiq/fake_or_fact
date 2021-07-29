import os
import pandas as pd
from sqlalchemy import create_engine
from flask import Flask, render_template, redirect, request, jsonify
#from config import postgrespwd
import re
import string
import numpy as np
#from tqdm import tqdm
import time
from collections import Counter
# from pathlib import Path
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
import nltk.data
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords

#db_string = f"postgresql://postgres:{postgrespwd}@localhost:5432/FakeNewsDetector"
#engine = create_engine(db_string)


app = Flask(__name__)
@app.route('/')
def welcome():
    return render_template("index.html")

@app.route('/verify/<subject>/<title>/<text>', methods=['GET', 'POST'])
def verifyArticle(subject,title,text):
    articleInfo = {'subject':subject,'title':title,'text':text}
    article_df = pd.DataFrame(articleInfo, index=[0])
    article_df['article'] = article_df['title']+" "+article_df['text']
    article_df.drop(['title','text'],axis=1,inplace=True)
    article_df['article'] = article_df['article'].str.replace('U.S.', 'USA').str.replace('U.S', 'USA').str.replace(' US ', ' USA ')
    @np.vectorize
    def wordpre(x):
        x = x.lower()
        x = re.sub('(?<!\w)([A-Za-z])\.', r'\1', x)
        x = re.sub('“|’|"|”', '', x)
        x = re.sub('\[.*?\]', '', x)
        x = re.sub("\\W"," ",x)
        x = re.sub('https?://\S+|www\.\S+', '', x)
        x = re.sub('<.*?>+', '', x)
        x = re.sub('[%s]' % re.escape(string.punctuation), '', x)
        x = re.sub('\n', '', x)
        x = re.sub('\w*\d\w*', '', x)
        return x
    article_df['article']= article_df['article'].apply(wordpre)
    stop = stopwords.words('english')
    article_df['article']= article_df['article'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    article_df['article_count'] = article_df['article'].apply(len)-1
    article_df['article_tokens'] = article_df['article'].apply(word_tokenize)
    article_df['article_tokens_count'] = article_df['article_tokens'].apply(len)
    subject_dummies = pd.get_dummies(article_df['subject'])
    ndf = pd.DataFrame()
    text = article_df['article'].apply(nltk.tokenize.WhitespaceTokenizer().tokenize)
    for i in text: 
        N = nltk.pos_tag(i)
        C = Counter([j for i,j in N])
        S = pd.Series([C])
        N = pd.DataFrame.from_records(S, columns = S.sum().keys())
        ndf = pd.concat([ndf, N], ignore_index=True, sort=False)
    ndf["sum"] = ndf.sum(axis=1)
    ndf = ndf.div(ndf['sum'], axis=0) *100
    features_df = pd.concat([
        article_df.drop(['subject','article','article_tokens'],axis=1),
        subject_dummies,
        ndf.drop('sum',axis=1)
        ], axis=1)
    # Feed features_df to the ML model
    return render_template("results.html",results=features_df.to_html())



if __name__ == "__main__":
    app.debug = True
    app.run()
