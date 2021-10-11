import pandas as pd
from matplotlib import pyplot as plt

def taskTwo():
    #2. Loads drug dataset to in Python to pandas dataframe using read_csv
    df = pd.read_csv('Datasets\Drugs\drug200.csv')

    #3 Plot distribution of instances of each class and saves to PDF
    df['Drug'].value_counts().plot(kind='bar')
    plt.savefig("Results//drug-distribution.pdf", dpi = 100)

    #4 Covert ordinal/nominal features to numerical
    df = pd.get_dummies(df.astype(str))
    #4 NOT FINISHED Still need to use categorical here

taskTwo()