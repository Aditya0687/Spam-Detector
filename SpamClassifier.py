import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle 

text_inp = input('Enter Message: ')

lemmatizer = WordNetLemmatizer()
corpus = []
review = re.sub('[^a-zA-Z]',' ',text_inp)
review = review.lower()
review = review.split()
review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
review = ' '.join(review)
corpus.append(review)

# load the model from disk
clf_best = pickle.load(open('finalized_model.sav', 'rb'))
cv = pickle.load(open('finalized_transformer.sav', 'rb'))

X_test_rt = cv.transform(corpus)

pred = clf_best.predict(X_test_rt)
if(pred==1):
    print('Spam')
else:
    print('Ham')