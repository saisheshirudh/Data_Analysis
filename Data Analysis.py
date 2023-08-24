# IMPORTING ALL THE NECESSARY LIBRARIES AND PACKAGES
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
%matplotlib inline
nltk.download('stopwords')



os.getcwd()
os.chdir('D:\\')

# LOADING THE DATASET AND SEEING THE DETAILS
data = pd.read_csv('yelp.csv')

# SHAPE OF THE DATASET
data.shape

# COLUMN NAMES
data.columns

# DATATYPE OF EACH COLUMN
data.dtypes

# SEEING FEW OF THE ENTRIES
data.head()

# DESCRIPTIVE STATISTICS
data.describe()



# CREATING A NEW COLUMN IN THE DATASET FOR THE NUMBER OF WORDS IN THE REVIEW
data['length'] = data['text'].apply(len)
data.head()



# PLOTTING BAR GRAPH TO SHOW CUSTOMER RATING
sns.countplot(x='stars', data = data)
plt.title("Count by Customer Rating")



# COMPARING TEXT LENGTH TO STARS
graph = sns.FacetGrid(data=data,col='stars')
graph.map(plt.hist,'length',bins=50,color='skyblue')



# GETTING THE MEAN VALUES OF THE VOTE COLUMNS WRT THE STARS ON THE REVIEW
stars = data.groupby('stars').mean()
stars



# FINDING THE CORRELATION BETWEEN THE VOTE COLUMNS
sns.heatmap(stars.corr(), cmap='PRGn')



# CLASSIFICATION
data_cls = data[(data['stars']==1) | (data['stars']==3) | (data['stars']==5)]
data_cls.head()
print(data_cls.shape)

# Seperate the dataset into X and Y for prediction
x = data_cls['text']
x.head()
y = data_cls['stars']
y.head()



# CLEANING THE REVIEWS - REMOVAL OF STOPWORDS AND PUNCTUATION
def txt_process(txt_pr):
    nopun = [char for char in txt_pr if char not in string.punctuation]
    nopun = ''.join(nopun)
    return [word for word in nopun.split() if word.lower() not in stopwords.words('english')]



# CONVERTING THE WORDS INTO A VECTOR
vec = CountVectorizer(analyzer=txt_process).fit(x)
print(len(vec.vocabulary_))
v0 = x[0]
print(v0)
vec0 = vec.transform([v0])
print(vec0)
print(vec.get_feature_names()[2687])
print(vec.get_feature_names()[12345])



x = vec.transform(x)
#Shape of the matrix:
print("Shape of the sparse matrix: ", x.shape)
#Non-zero occurences:
print("Non-Zero occurences: ",x.nnz)



# SPLITTING THE DATASET INTO TRAINING SET AND TESTING SET
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=101)



random.seed(25)

# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
multi_nb = MultinomialNB()
multi_nb.fit(x_train,y_train)
pred_mnb = multi_nb.predict(x_test)
print("Confusion Matrix for Multinomial Naive Bayes:")
print(confusion_matrix(y_test,pred_mnb))
mnb_score = round(accuracy_score(y_test,pred_mnb)*100,2)
print("Score:", mnb_score)
print("Classification Report:",classification_report(y_test,pred_mnb))



random.seed(25)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
pred_rf = rf.predict(x_test)
print("Confusion Matrix for Random Forest Classifier:")
print(confusion_matrix(y_test,pred_rf))
rf_score = round(accuracy_score(y_test,pred_rf)*100,2)
print("Score:", rf_score)
print("Classification Report:",classification_report(y_test,pred_rf))



random.seed(25)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
pred_dt = dt.predict(x_test)
print("Confusion Matrix for Decision Tree:")
print(confusion_matrix(y_test,pred_dt))
dt_score = round(accuracy_score(y_test,pred_dt)*100,2)
print("Score:", dt_score)
print("Classification Report:",classification_report(y_test,pred_dt))



random.seed(25)

# Support Vector Machine
from sklearn.svm import SVC
svm = SVC(random_state=101)
svm.fit(x_train,y_train)
pred_svm = svm.predict(x_test)
print("Confusion Matrix for Support Vector Machines:")
print(confusion_matrix(y_test,pred_svm))
svm_score = round(accuracy_score(y_test,pred_svm)*100,2)
print("Score:", svm_score)
print("Classification Report:",classification_report(y_test,pred_svm))



random.seed(25)

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(learning_rate=0.1,max_depth=5,max_features=0.5,random_state=999999)
gbc.fit(x_train,y_train)
pred_gbc = gbc.predict(x_test)
print("Confusion Matrix for Gradient Boosting Classifier:")
print(confusion_matrix(y_test,pred_gbc))
gbc_score = round(accuracy_score(y_test,pred_gbc)*100,2)
print("Score:", gbc_score)
print("Classification Report:",classification_report(y_test,pred_gbc))



random.seed(25)

# K Nearest Neighbour Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
pred_knn = knn.predict(x_test)
print("Confusion Matrix for K Neighbors Classifier:")
print(confusion_matrix(y_test,pred_knn))
knn_score = round(accuracy_score(y_test,pred_knn)*100,2)
print("Score: ", knn_score)
print("Classification Report:")
print(classification_report(y_test,pred_knn))



random.seed(25)

# MULTILAYER PERCEPTRON CLASSIFIER
from sklearn.neural_network import MLPClassifier
mpc = MLPClassifier()
mpc.fit(x_train,y_train)
pred_mpc = mpc.predict(x_test)
print("Confusion Matrix for Multilayer Perceptron Classifier:")
print(confusion_matrix(y_test,pred_mpc))
mpc_score = round(accuracy_score(y_test,pred_mpc)*100,2)
print("Score:", mpc_score)
print("Classification Report:")
print(classification_report(y_test,pred_mpc))



# COMPARISON OF ACCURACY BETWEEN MACHINE LEARNING MODELS AND CHOOSING BEST MODEL TO TRAIN THE TEST DATA
print('Accuracy with Multinomial Naive Bayes:', mnb_score)
print('Accuracy with Random Forest:', rf_score)
print('Accuracy with Decision Tree:', dt_score)
print('Accuracy with Support Vector Machine:', svm_score)
print('Accuracy with Gradient Boosting Classifier:', gbc_score)
print('Accuracy with K Nearest Neighbour Algorithm:', knn_score)
print('Accuracy with Multilayer perceptron classifier:', mpc_score)



# POSITIVE REVIEW
pos = data['text'][0]
print(pos)
print("Actual Rating: ",data['stars'][0])
pos_t = vec.transform([pos])
print("Predicted Rating:")
mpc.predict(pos_t)[0]



# AVERAGE REVIEW
avg = data['text'][16]
print(avg)
print("Actual Rating: ",data['stars'][16])
avg_t = vec.transform([avg])
print("Predicted Rating:")
mpc.predict(avg_t)[0]



# NEGATIVE REVIEW
neg = data['text'][16]
print(neg)
print("Actual Rating: ",data['stars'][23])
neg_t = vec.transform([neg])
print("Predicted Rating:")
mpc.predict(neg_t)[0]





