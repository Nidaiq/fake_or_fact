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
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pickle
from sklearn.preprocessing import StandardScaler

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
    article_df['title'] = article_df['title'].str.replace('U.S.', 'USA').str.replace('U.S', 'USA').str.replace(' US ', ' USA ')
    article_df['text'] = article_df['text'].str.replace('U.S.', 'USA').str.replace('U.S', 'USA').str.replace(' US ', ' USA ')
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
    article_df['title']= article_df['title'].apply(wordpre)
    article_df['text']= article_df['text'].apply(wordpre)
    article_df['article']= article_df['article'].apply(wordpre)
    stop = stopwords.words('english')
    article_df['title']= article_df['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    article_df['text']= article_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    article_df['title_count'] = article_df['title'].apply(len)-1
    article_df['text_count'] = article_df['text'].apply(len)-1
    article_df['title_tokens'] = article_df['title'].apply(word_tokenize)
    article_df['text_tokens'] = article_df['text'].apply(word_tokenize)
    article_df['title_tokenized_count'] = article_df['title_tokens'].apply(len)
    article_df['text_tokenized_count'] = article_df['text_tokens'].apply(len)
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
    article_df.drop(['subject','title','text','article','title_tokens','text_tokens'],axis=1),
    subject_dummies,
    ndf.drop('sum',axis=1)
    ], axis=1)
    reqd_features = ['title_count', 'text_count', 'title_tokenized_count',
        'text_tokenized_count', 'US News', 'World News', 'JJ', 'NN', 'VBZ',
        'RP', 'VBG', 'VBP', 'DT', 'RB', 'VB', 'CC', 'PRP', 'IN', 'VBD', 'TO',
        'PRP$', 'NNS', 'JJS', 'CD', 'JJR', 'RBR', 'VBN', 'MD', 'WP', 'FW',
        'NNP', 'WRB', 'WDT', 'PDT', 'EX', 'RBS', 'NNPS', 'UH', 'WP$', 'POS']
    for i in reqd_features:
        if(i not in list(features_df.columns)):
            features_df[i]=0
    features = features_df.iloc[0]
    root = os.path.dirname(os.path.abspath(__file__))  
    ml_file_path = os.path.join(root, 'static/svm_model.sav')
    model = pickle.load(open(ml_file_path, 'rb'))
    result = model.predict(features)[0]
    
    return render_template("results.html",results=result)



if __name__ == "__main__":
    app.debug = True
    app.run()
