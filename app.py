import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from flask import Flask, request, render_template
from flask_cors import cross_origin


app = Flask(__name__)
# load the model from disk
clf_best = pickle.load(open('finalized_model.sav', 'rb'))
cv = pickle.load(open('finalized_transformer.sav', 'rb'))


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        text_inp = request.form["text_msg"]
        lemmatizer = WordNetLemmatizer()
        corpus = []
        review = re.sub('[^a-zA-Z]', ' ', text_inp)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)

        X_test_rt = cv.transform(corpus)
        pred = clf_best.predict(X_test_rt)
        output = ''
        if (pred == 1):
            output = 'Spam'
        else:
            output = 'Ham'
        return render_template('home.html', prediction_text="Given message is a {}".format(output))
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)