# Importing the required libraries
from matplotlib import pyplot as plt
import numpy as np
#import os package to use file related methods
import os
#initialization of file directories
def plot_bbc_groups():
    directories = ["Datasets\\BBC\\business", "Datasets\\BBC\\entertainment", "Datasets\\BBC\\politics", "Datasets\\BBC\\sport" ,"Datasets\\BBC\\tech"]
    categories = ["Business", "Entertainment", "Politics", "Sport", "Technology"]
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

plot_bbc_groups()