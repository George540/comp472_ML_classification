
from matplotlib import pyplot as plt
import numpy as np
import os
from numpy.core.numeric import full
from sklearn import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

directories = ["Datasets\\BBC\\business", "Datasets\\BBC\\entertainment", "Datasets\\BBC\\politics", "Datasets\\BBC\\sport" ,"Datasets\\BBC\\tech"]
categories = ["Business", "Entertainment", "Politics", "Sport", "Technology"]

def plot_bbc_groups():
    '''
    TASK 1 - Question 2
    2 - Plot the distribution of the instances in each class and save the graphic in a file called BBC-distribution.pdf.
    '''
    nbrFilesPerCtgry = []
    for directory in directories:
        dirListing = os.listdir(directory)
        nbrFilesPerCtgry.append(len(dirListing))
    plt.bar(categories, nbrFilesPerCtgry, width=0.7, bottom=50)
    plt.xlabel('Categories', fontsize=15)
    plt.ylabel('Files', fontsize=15)
    for i in range(len(categories)):
        plt.text(i, nbrFilesPerCtgry[i], nbrFilesPerCtgry[i], ha='center', va='bottom')
    plt.show()
    plt.savefig("Results//BBC-distribution.pdf", dpi = 100)

def assign_category_name():
    '''
    TASK 1 - Question 3-4-5
    3 - Loads the corpus
    4 - Pre-processes the dataset to have the features ready to be used by multinomial Naive Bayes classifier
    5 - Splits the dataset into 80% for training and 20% for testing
    '''
    bbc_loaded_files = datasets.load_files("Datasets\\BBC", encoding = "latin1")
    vectorizer = CountVectorizer()
    full_data_set = vectorizer.fit_transform(bbc_loaded_files.values) #X value
    
    #print(full_data_set.data) #frequency
    #print(full_data_set.indices) #position in matrix
    #print(full_data_set.data)

    #print(list(zip(vectorizer.get_feature_names(), full_data_set.sum(0).getA1())))
    #X_training_set, X_testing_set, y_training_Set, y_testing_set = train_test_split(full_data_set.data, categories, test_size=0.2, random_state = None)
    #print(training_set.shape[0])
    #print(testing_set.shape[0])
    #clf = MultinomialNB()
    #clf.fit(training_set, testing_set)
    #MultinomialNB()
    #print(clf.predict(testing_set))
    
#plot_bbc_groups()
assign_category_name()

