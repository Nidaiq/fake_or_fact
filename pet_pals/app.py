from flask import Flask, render_template
import pandas as pd
import os
import pickle

# db_string = f"postgresql://postgres:{postgrespwd}@localhost:5432/FakeNewsDetector"
# engine = create_engine(db_string)

app = Flask(__name__)
@app.route('/')
def welcome():
    return render_template("index.html")

@app.route('/verdict/<subject>/<title>/<text>', methods=['GET', 'POST'])
def classifyArticle(subject,title,text):
    
    # Read Test Data into a Dataframe
    articleInfo = {'subject':subject,'title':title,'text':text}
    article_df = pd.DataFrame(articleInfo, index=[0])

    # Load ML Dependencies
    root = os.path.dirname(os.path.abspath(__file__)) 
    cv_file_path = os.path.join(root,'static/machineLearning/cv.sav')
    nb_file_path = os.path.join(root,'static/machineLearning/nb_model.sav')
    countVectorizer = pickle.load(open(cv_file_path,'rb'))
    naiveBayes = pickle.load(open(nb_file_path,'rb'))

    # Predict Outcome
    cv_text = countVectorizer.transform(article_df['text'])
    outcome = naiveBayes.predict(cv_text)[0]
    if (outcome == 0):
        classification = 'fact'
    elif (outcome == 1):
        classification = 'fake'

    return render_template("results.html",results=classification)

if __name__ == "__main__":
    app.run(debug=True)