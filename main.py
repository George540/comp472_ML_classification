
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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
    'Question 3'
    bbc_loaded_files = load_files("Datasets\\BBC", encoding = "latin1")
    #verify the count of each category and get the target names (categories) and store as 'labels_str' now labels is value 0-4
    labels, counts = np.unique(bbc_loaded_files.target, return_counts=True)
    labels_str = np.array(bbc_loaded_files.target_names)[labels]
    print(labels_str,counts)
    'Question 5'
    #bbc_loadfiles.data is the data itself
    #bbc_loadfiles.target is the labels/categories
    X_train, X_test, Y_train, Y_test = train_test_split(bbc_loaded_files.data, bbc_loaded_files.target, test_size=0.2, random_state = None)
    #print('80%:',len(X_train),'20%:',len(X_test))
    'Question 4'
    #it is best to use only the testing set to form our matrix
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)
    'Question 6/7a'
    clf = MultinomialNB()
    #transform the data to document term matrix and pass labels
    clf.fit(vectorizer.transform(X_train) , Y_train)
    #print(clf.predict(testing_set))
    Y_pred = clf.predict(vectorizer.transform(X_test))
    'Question 7'
    #print(confusion_matrix(Y_test, Y_pred)) #b
    #print(classification_report(Y_test, Y_pred)) #c
    #print(accuracy_score(Y_test, Y_pred)) #d
    #print(labels_str, counts/len(bbc_loaded_files.target)) #e
    #print(len(vectorizer.get_feature_names())) #f

    #7g
    topics_word_count = []
    for topic in bbc_loaded_files.target_names:
        topic_list = []
        topic_list.append(topic)
        topics_word_count.append(load_files_by_category(topic_list))

    print(topics_word_count)

    #7h
    total_words = 0;
    word_frequency = vectorizer.fit_transform(bbc_loaded_files.data).toarray().sum(axis=0)
    for x in range(0, len(word_frequency)):
        total_words += word_frequency[x]
    print(total_words)

def load_files_by_category(c):
    category_loaded_files = load_files("Datasets\\BBC", categories = c, encoding = "latin1")
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(category_loaded_files.data)
    return len(vectorizer.get_feature_names())

#plot_bbc_groups()
assign_category_name()

