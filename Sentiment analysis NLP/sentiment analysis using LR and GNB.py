"""
Created on Thru Nov 23 22:16:36 2020

@author: mahesh
"""

#----------------------------- Natural Language Processing - Logistic regression and GaussianNaiveBayes----------------

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
corpus = []
ps = WordNetLemmatizer()
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 177013)

# training the Naive Bayes model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred_NB = classifier.predict(X_test)

# training Logistic regression 
LR = LogisticRegression(solver = "lbfgs")
LR.fit(X_train,y_train)
y_pred_LR = LR.predict(X_test)

# Making the Confusion Matrix and accuracy
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score
cm_nb = confusion_matrix(y_test, y_pred_NB)
cm_lr = confusion_matrix(y_test, y_pred_LR)

print(f"\nlogistic regression accuracy : {accuracy_score(y_test,y_pred_LR)}\n {metrics.classification_report(y_test,y_pred_LR):.>{20}}\nGaussian Naive Bayes Classifier accuracy : {accuracy_score(y_test,y_pred_NB)} \n{metrics.classification_report(y_test,y_pred_NB):.>{8}}")

