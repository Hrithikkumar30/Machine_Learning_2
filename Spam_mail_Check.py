import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  #Because machine learning models can not understand the text data sets, this package will change text data to numerical data

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data Collecting and Pre-Processing

raw_mail_dataset = pd.read_csv("C:/Users/alfa/Desktop/machineLearning/SPAM_MAIL_CHEcK/mail_data.csv")
#print(raw_mail_dataset)

#Replace null values with null strings
mail_dataset = raw_mail_dataset.where((pd.notnull(raw_mail_dataset)), '')
#print(mail_dataset)

#printing the first 5 rows of the dataframe
#print(mail_dataset.head())

#Label Encoding 

#Label spam mail as 0;  Not_spam mail as 1;

mail_dataset.loc[mail_dataset['Category']=='spam','Category',]= 0    #spam datas will considered as 0
mail_dataset.loc[mail_dataset["Category"]=="Not_spam","Category",]=  1 #Not_spam datas will considered as 1

#seperating the datasets into texts and label

X = mail_dataset['Message'] 
Y = mail_dataset['Category']
#print(X)
#print(Y)

# Spliting the data into training and   data
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.2,random_state =5)
#print(X.shape)
#print(X_train.shape)
#print(Y_test.shape)

#feature_extraction

#Transforming the test data to feature vectors that can used as input to the logistic regression model
feature_extraction = TfidfVectorizer(min_df =1, stop_words = "english",lowercase = 'True')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train =Y_train.astype('int')
Y_test =Y_test.astype('int')

#print(X_train_features) 

#Training and testing model by Logistic Regression
model = LogisticRegression()
model.fit(X_train_features,Y_train)

#prediction on training data
Prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, Prediction_on_training_data)
#print(accuracy_on_training_data)

#prediction on testing data
Prediction_on_testing_data = model.predict(X_test_features)
accuracy_on_testing_data = accuracy_score(Y_test, Prediction_on_testing_data)
#print(accuracy_on_testing_data)

#BUILDING A PREDICTIVE SYSTEM

input_mail = [",Wow. I never realized that you were so embarassed by your accomodations. I thought you liked it, since i was doing the best i could and you always seemed so happy about ""the cave"". I'm sorry I didn't and don't have more to give. I'm sorry i offered. I'm sorry your room was so embarassing."]
#convert text to feature vectors
input_mail_feature = feature_extraction.transform(input_mail)
prediction = model.predict(input_mail_feature)
#print(prediction)

if prediction[0]== "Not_spam":
    print("It is Not a Spam Mail ")
else:
    print("Ignore, It is a Spam Mail")