
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

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
    doc_matrix = vectorizer.transform(X_train)
    #print(vectorizer.get_feature_names())
    'Question 6/7a'
    clf = MultinomialNB()
    #transform the data to document term matrix and pass labels
    clf.fit(doc_matrix, Y_train)
    Y_pred = clf.predict(vectorizer.transform(X_test))
    #print(clf.feature_count_)
    #TODO turn above in to variable and iterate to count word count per row.
    'Question 7'
    #print(confusion_matrix(Y_test, Y_pred)) #b
    #print(classification_report(Y_test, Y_pred)) #c
    #print(accuracy_score(Y_test, Y_pred)) #d
    print('--------------------------------------------------------------------------------------------')
    print(f1_score(Y_test, Y_pred, average='macro'))#d pt2
    print(f1_score(Y_test, Y_pred, average='weighted')) #d pt3
    print(labels_str, counts/len(bbc_loaded_files.target)) #e
    print(len(vectorizer.vocabulary_)) #f the total number of words in the vocabulary
    print('-------------------------------------------G/H/I/J-------------------------------------------')
    #7g
    print(clf.classes_)
    word_count_per_label = 0
    total_words_in_corpus = 0
    total_one_count_words_in_corpus = 0
    matrix_array = clf.feature_count_
    zero_count_for_this_label = 0
    one_count_for_this_label = 0
    curr_lbl = 0
    for label in matrix_array:
        #label prints out the array of a label/category
        print(curr_lbl,': ',label)
        for count_for_this_word in label:
            if count_for_this_word == 0:
                zero_count_for_this_label = zero_count_for_this_label +1
            elif count_for_this_word == 1:
                one_count_for_this_label = one_count_for_this_label + 1
                word_count_per_label = word_count_per_label + count_for_this_word
            else:
                word_count_per_label = word_count_per_label + count_for_this_word
        print('Word count for this label: ',word_count_per_label) #g
        print('Number of zero counts per word found in this label: ', zero_count_for_this_label, ' Percentage: ', round(((zero_count_for_this_label/word_count_per_label)*100),2),'%')#I
        total_words_in_corpus = word_count_per_label + total_words_in_corpus
        total_one_count_words_in_corpus = one_count_for_this_label + total_one_count_words_in_corpus
        zero_count_for_this_label = 0
        word_count_per_label = 0
        curr_lbl = curr_lbl + 1
    print('\ntotal words in all folders/files: ', total_words_in_corpus)#h
    print('Number of singular words in corpus: ', total_one_count_words_in_corpus, 'Percentage: ',round(((total_one_count_words_in_corpus/total_words_in_corpus)*100),2),'%')#J
    print('--------------------------------------------------------------------------------------------')
    docmatrix_toarray = doc_matrix.toarray()
    test = pd.DataFrame(docmatrix_toarray, columns=vectorizer.get_feature_names())
    print(test)


#plot_bbc_groups()
assign_category_name()

