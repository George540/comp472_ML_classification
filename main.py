# Importing the required libraries
from matplotlib import pyplot as plt
import numpy as np
#import os package to use file related methods
import os

from sklearn import *

directories = ["Datasets\\BBC\\business", "Datasets\\BBC\\entertainment", "Datasets\\BBC\\politics", "Datasets\\BBC\\sport" ,"Datasets\\BBC\\tech"]
categories = ["Business", "Entertainment", "Politics", "Sport", "Technology"]

#initialization of file directories
def plot_bbc_groups():
    nbrFilesPerCtgry = []

    for directory in directories:
        dirListing = os.listdir(directory)
        nbrFilesPerCtgry.append(len(dirListing))

    # Creating a bar chart with the parameters
    plt.bar(categories, nbrFilesPerCtgry, width=0.7, bottom=50)
    plt.xlabel('Categories', fontsize=15)
    plt.ylabel('Files', fontsize=15)
    for i in range(len(categories)):
        plt.text(i, nbrFilesPerCtgry[i], nbrFilesPerCtgry[i], ha='center', va='bottom')
    #plt.show()
    plt.savefig("Results//BBC-distribution.pdf", dpi = 100)

def assign_category_name():

    data = datasets.load_files("Datasets\\BBC", encoding = "latin1")

    vectorized = feature_extraction.text.CountVectorizer(input='data')
    matrix = vectorized.fit_transform(data)
    print(vectorized.vocabulary)
    
    


#plot_bbc_groups()
assign_category_name()

