
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn import *
from sklearn.feature_extraction.text import CountVectorizer

directories = ["Datasets\\BBC\\business", "Datasets\\BBC\\entertainment", "Datasets\\BBC\\politics", "Datasets\\BBC\\sport" ,"Datasets\\BBC\\tech"]
categories = ["Business", "Entertainment", "Politics", "Sport", "Technology"]

def plot_bbc_groups():
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
    text_train_subset = datasets.load_files("Datasets\\red", encoding = "latin1")
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(text_train_subset.data)
    print(list(zip(vectorizer.get_feature_names(), X_train.sum(0).getA1())))
    
#plot_bbc_groups()
assign_category_name()

