##########################################
## taskOne.py (Mini-Assignment 1 COMP 472)
## This code contains the execution of all operations for Task 1
## Created by Team Oranges
##########################################

from matplotlib import pyplot as plt
import numpy as np
import os
import os.path
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

# TASK 1 - Question 3
'Load the files using load_files()'

bbc_performance_file_name = 'Results\\bbc-performance.txt'
directories = ["Datasets\\BBC\\business", "Datasets\\BBC\\entertainment", "Datasets\\BBC\\politics", "Datasets\\BBC\\sport" ,"Datasets\\BBC\\tech"]
categories = ["Business", "Entertainment", "Politics", "Sport", "Technology"]
bbc_loaded_files = load_files("Datasets\\BBC", encoding = "latin1")
f = open(bbc_performance_file_name, "w")

# TASK 1 - Question 5
'Split data set using train_test_split'
'bbc_loadfiles.data is the data itself, X'
'bbc_loadfiles.target is the labels/categories, y'

X_train, X_test, Y_train, Y_test = train_test_split(bbc_loaded_files.data, bbc_loaded_files.target, test_size=0.2, shuffle=False)

# TASK 1 - Question 4
'Preprocess data set with CountVectorizer()'
vectorizer = CountVectorizer()
vectorizer.fit(X_train)
doc_matrix = vectorizer.transform(X_train)
#label and feature information
labels, counts = np.unique(bbc_loaded_files.target, return_counts=True)
labels_str = np.array(bbc_loaded_files.target_names)[labels]

def plot_bbc_groups():
    # TASK 1 - Question 2
    'Plot the distribution of the instances in each class and save the graphic in a file called BBC-distribution.pdf'
    nbrFilesPerCtgry = []
    for directory in directories:
        dirListing = os.listdir(directory)
        nbrFilesPerCtgry.append(len(dirListing))
    plt.bar(categories, nbrFilesPerCtgry, width=0.7, bottom=50)
    plt.xlabel('Categories', fontsize=15)
    plt.ylabel('Files', fontsize=15)
    for i in range(len(categories)):
        plt.text(i, nbrFilesPerCtgry[i], nbrFilesPerCtgry[i], ha='center', va='bottom')
    plt.savefig("Results//BBC-distribution.pdf", dpi = 100)

def assign_category_name(bbc_loaded_files,smoothing):
    #Question 6/
    'Usage of the MultinomialNB to train and predict Y_test'
    #clf stands for classifier
    clf = MultinomialNB(alpha=smoothing)
    #pass labels and document term matrix
    clf.fit(doc_matrix, Y_train)
    Y_pred = clf.predict(vectorizer.transform(X_test))
    
    #Write reports to text file
    f.write('Confusion Matrix: \n')
    f.write(str(confusion_matrix(Y_test, Y_pred)))
    f.write('\nClassification Report: \n')
    f.write(str(classification_report(Y_test, Y_pred)))
    f.write('\nAccuracy Score: \n')
    f.write(str(accuracy_score(Y_test, Y_pred))+'\n')
    f.write('\nF1 score, macro: \n')
    f.write(str(f1_score(Y_test, Y_pred, average='macro'))+'\n')
    f.write('\nF1 score, weighted: \n')
    f.write(str(f1_score(Y_test, Y_pred, average='weighted'))+'\n')
    f.write('\nPrior probability of each class: \n')
    f.write(str(labels_str))
    f.write(' '+str(counts/len(bbc_loaded_files.target))+'\n') 
    f.write('\nTotal Number of words in the vocabulary: \n')
    f.write(str(len(vectorizer.vocabulary_)))

    #7g

    word_count_per_label = 0
    total_words_in_X_train = 0
    total_one_count_words_in_X_train = 0
    matrix_array = clf.feature_count_
    zero_count_for_this_label = 0
    one_count_for_this_label = 0
    curr_lbl = 0
    favorite_words = ['gaming', 'legendary']
    f.write("\nGathering numerical data per label...\n")
    for label in matrix_array:
        #label prints out the array of a label/category
        f.write(str(labels_str[curr_lbl]) + ': ' +str(label)+'\n')
        for count_for_this_word in label:
            if count_for_this_word == 0:
                zero_count_for_this_label = zero_count_for_this_label +1
            elif count_for_this_word == 1:
                one_count_for_this_label = one_count_for_this_label + 1
                word_count_per_label = word_count_per_label + count_for_this_word
            else:
                word_count_per_label = word_count_per_label + count_for_this_word
        #g the number of word-tokens in each class
        ' the number of word-tokens in each class (i.e. the number of words in total)'
        f.write(str(labels_str[curr_lbl]) + ':G, Word count for this label: '+ str(word_count_per_label)+'\n')
        #I the number and percentage of words with a frequency of zero in each class
        'the number and percentage of words with a frequency of zero in each class'
        f.write(str(labels_str[curr_lbl])+':I, Number of zero counts per word found in this label: '+ str(zero_count_for_this_label)+ ' Percentage: '+ str(round(((zero_count_for_this_label/word_count_per_label)*100),2))+'%\n')
        f.write('-------------------\n')
        total_words_in_X_train = word_count_per_label + total_words_in_X_train
        total_one_count_words_in_X_train = one_count_for_this_label + total_one_count_words_in_X_train
        zero_count_for_this_label = 0
        word_count_per_label = 0
        curr_lbl = curr_lbl + 1
    #h
    'the number of word-tokens in the entire corpus'
    f.write('\nH, total words in all folders/files: '+ str(total_words_in_X_train))
    
    #J
    'the number and percentage of words with a frequency of one in the entire corpus'
    f.write('\nJ, Number of singular words in X_ train corpus: '+ str(total_one_count_words_in_X_train) + 'Percentage: '+str(round(((total_one_count_words_in_X_train/total_words_in_X_train)*100),2))+'%')
    f.write('\n'+"-"*58+'\n')
    
    #7k
    'your 2 favorite words (that are present in the vocabulary) and their log-prob'
    probabilities = clf.predict_log_proba(vectorizer.transform(favorite_words))
    f.write(str(probabilities))
    for col, label in enumerate(categories):
        for row, word in enumerate(favorite_words):
            f.write('\n[' +str(label)+' ] Log-Prob of word \''+str(word.upper())+'\':'+str(probabilities[row][col]))

def runTaskOne():
    plot_bbc_groups()
    #7a
    #print("-"*20,"try 1","-"*20)
    f.write("----------try 1----------\n\n")
    assign_category_name(bbc_loaded_files, 1.0)
    #8
    #try 2
    f.write("\n\n----------try 2----------\n\n")
    assign_category_name(bbc_loaded_files, 1.0)
    #9
    #Smoothing 0.0001
    f.write("\n\n----------Smoothing 0.0001----------\n\n")
    assign_category_name(bbc_loaded_files, 0.0001)
    #10
    #Smoothing 0.9
    f.write("\n\n----------Smoothing 0.9----------\n\n")
    assign_category_name(bbc_loaded_files, 0.9)
