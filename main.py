## importing the dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

## data collection & preprocessing

# laoding the data from csv file to pandas dataframe
raw_mail_data = pd.read_csv('mail_data.csv')
# print(raw_mail_data.head())

# replace null values with a null string
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')
# print(mail_data.head())

# checking the number of rows and columns in the dataframe
# print(mail_data.shape, '\n')

# label encoding

# label spam mail as 0 and ham mail as 1

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

## separating the data as texts ans labels

x = mail_data['Message']
y = mail_data['Category']

# print(x)
# print(y)

# splitting the data into training and testing data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)

## feature extraction

# transform the text data to feature vectors that can be used as input for the logistic regression model
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

# convert y_train and y_test as integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# print(x_train_features)

# training the model

## logistic regression model

model = LogisticRegression()

# training the logistic regression model with the training data
model.fit(x_train_features, y_train)


## evaluating the trained model

# prediction on training data
prediction_on_training_data = model.predict(x_train_features)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)

# print('Accuracy on training data : ', accuracy_on_training_data)

# prediction on test data

prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)

# print('Accuracy on test data : ', accuracy_on_test_data)

# building a predictive system
input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times."]

# convert text to feature vectors
input_mail_features = feature_extraction.transform(input_mail)


## making predictions
prediction = model.predict(input_mail_features)  
print(prediction)           


if prediction[0] == 0:
    print('The mail is a spam mail')
else:
    print('The mail is a ham mail')
