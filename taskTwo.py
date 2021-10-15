from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import os.path
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

def taskTwo():
    #2. Loads drug dataset to in Python to pandas dataframe using read_csv
    df = pd.read_csv('Datasets\Drugs\drug200.csv')

    #3 Plot distribution of instances of each class and saves to PDF
    df['Drug'].value_counts().plot(kind='bar')
    plt.savefig("Results//drug-distribution.pdf", dpi = 100)

    #4 Covert all ordinal and nominal features to numerical format
    '''
    get_dummies is used to turn nominal values into 1-0s
    Categorical is used to turn ordinal values in ordered ranges from 0-2. (ie High = 2, Normal = 1, Low = 1)
    '''
    temp_df = pd.get_dummies(df['Sex'])
    df = df.drop('Sex', axis = 1)
    df = df.join(temp_df)
    #temp_df = pd.get_dummies(df['Drug'])
    #df = df.drop('Drug', axis = 1)
    #df = df.join(temp_df)
    df['BP'] = pd.Categorical(df['BP'], ordered = True, categories=['LOW', 'NORMAL', 'HIGH'])
    df['BP'] = df['BP'].cat.codes
    df['Cholesterol'] = pd.Categorical(df['Cholesterol'], ordered = True, categories=['LOW', 'NORMAL', 'HIGH'])
    df['Cholesterol'] = df['Cholesterol'].cat.codes

    #5 Split dataset using train_test_split using the default parameter values
    '''
    Because we are using a dataframe we need to extract the label Drug in order to use train_test_split()
    '''
    y = df.pop('Drug')
    X = df
    X_train,X_test,y_train,y_test = train_test_split(X.index,y ,test_size=0.2)
    
    #6 Run 6 different classifier
    #6-a) Gaussian Naive Bayes Classifier
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))


taskTwo()