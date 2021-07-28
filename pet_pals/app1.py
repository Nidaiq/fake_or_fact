# import necessary libraries
import os
import pandas as pd
from sqlalchemy import create_engine
#from config import postgrespwd
import re
import string
import numpy as np
#from tqdm import tqdm
import time
from collections import Counter
import nltk.data
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

#################################################
# Database Setup
#################################################

from flask_sqlalchemy import SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', '') or "sqlite:///db.sqlite"

# Remove tracking modifications
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

from .models import Pet


# create route that renders index.html template
@app.route("/")
def welcome():
    return render_template("index.html")


# Query the database and send the jsonified results

@app.route('/verify/<title>/<text>', methods=['GET', 'POST'])
def verifyArticle(title,text):
    articleInfo = {'title':title,'text':text}
    article_df = pd.DataFrame(articleInfo, index=[0])
    article_df['article'] = article_df['title']+" "+article_df['text']
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
    nltk.download('averaged_perceptron_tagger')
    ndf = pd.DataFrame()
    text = article_df['article'].apply(nltk.tokenize.WhitespaceTokenizer().tokenize)
    for i in tqdm(text): 
        N = nltk.pos_tag(i)
        C = Counter([j for i,j in N])
        S = pd.Series([C])
        N = pd.DataFrame.from_records(S, columns = S.sum().keys())
        ndf = pd.concat([ndf, N], ignore_index=True, sort=False)
    ndf = ndf.fillna(0)
    ndf["sum"] = ndf.sum(axis=1)
    normalized_df = ndf.div(ndf['sum'], axis=0) *100
    nlp_output_df = pd.concat([article_df,normalized_df], axis=1)
    nlp_output_df = nlp_output_df.drop(['text','title','article'], axis = 1)
    # Output from ML is returned in next line
    return render_template("results.html",results=nlp_output_df.to_html())
    # return(articleInfo)

if __name__ == "__main__":
	app.debug = True
app.run()
