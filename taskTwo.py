from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import os.path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

drug_performance_file_name = 'Results\\drug-performance.txt'
NB_results = {'accuracy-score-sum': [], 'f1-score-macro-sum': [] , 'f1-score-weighted-sum': []}
Base_DT_results = {'accuracy-score-sum': [], 'f1-score-macro-sum': [] , 'f1-score-weighted-sum': []}
Top_DT_results = {'accuracy-score-sum': [], 'f1-score-macro-sum': [] , 'f1-score-weighted-sum': []}
Per_results = {'accuracy-score-sum': [], 'f1-score-macro-sum': [] , 'f1-score-weighted-sum': []}
Base_MLP_results = {'accuracy-score-sum': [], 'f1-score-macro-sum': [], 'f1-score-weighted-sum': []}
Top_MLP_results = {'accuracy-score-sum': [], 'f1-score-macro-sum': [], 'f1-score-weighted-sum': []}
write_file_flag = True

def textfile_generator():
    
    # Function verifies if bbc-performance.txt exists
    # IF file exists it will increment 1 to the name of the file
    # I.E. drug-performance1.txt, drug-performance2.txt, drug-performance3.txt ect.. 
    
    count = 1
    global drug_performance_file_name
    while os.path.isfile(drug_performance_file_name) :
        drug_performance_file_name = 'Results\\drug-performance'+str(count)+'.txt'
        count +=1

def taskTwo():
    #2. Loads drug dataset to in Python to pandas dataframe using read_csv
    df = pd.read_csv('Datasets\Drugs\drug200.csv', dtype=str)

    #3 Plot distribution of instances of each class and saves to PDF
    #df['Drug'].value_counts().plot(kind='bar')
    #plt.savefig("Results//drug-distribution.pdf", dpi = 100)

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
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    vectorizer = CountVectorizer(lowercase=False)
    vectorizer.fit(X_train)
    doc_matrix = vectorizer.transform(X_train)

    #6 Run 6 different classifier
    #6-a) Gaussian Naive Bayes Classifier
    asterisks = "-"*20
    f.write("\n---------------------------------------- \n")
    print("\n1) GaussianNB\n")
    f.write("\n1) GaussianNB\n")
    gnb = GaussianNB()
    y_pred_nb = gnb.fit(X_train, y_train).predict(X_test)
    # print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    temp = print_reports(y_test, y_pred_nb)
    NB_results['accuracy-score-sum'].append(temp['accuracy-score'])
    NB_results['f1-score-macro-sum'].append(temp['f1-score-macro'])
    NB_results['f1-score-weighted-sum'].append(temp['f1-score-weighted'])

    print("----------------------------------------")
    f.write("\n----------------------------------------\n")
    print("\n2) Base-DT\n")
    f.write("\n2) Base-DT\n")
    bdt = DecisionTreeClassifier()
    y_pred_bdt = bdt.fit(X_train, y_train).predict(X_test)
    print_reports(y_test, y_pred_bdt)
    temp = print_reports(y_test, y_pred_bdt)
    Base_DT_results['accuracy-score-sum'].append(temp['accuracy-score'])
    Base_DT_results['f1-score-macro-sum'].append(temp['f1-score-macro'])
    Base_DT_results['f1-score-weighted-sum'].append(temp['f1-score-weighted'])

    print("----------------------------------------")
    f.write("\n----------------------------------------\n")
    print("\n3) Top-DT\n")
    f.write("\n3) Top-DT\n")
    parameters = {'criterion':['gini', 'entropy'], 'max_depth':[4,5], 'min_samples_split':[4,5,6]}
    gsc = GridSearchCV(DecisionTreeClassifier(), parameters)
    y_pred_tdt = gsc.fit(X_train, y_train).predict(X_test)
    temp = print_reports(y_test, y_pred_tdt)
    Top_DT_results['accuracy-score-sum'].append(temp['accuracy-score'])
    Top_DT_results['f1-score-macro-sum'].append(temp['f1-score-macro'])
    Top_DT_results['f1-score-weighted-sum'].append(temp['f1-score-weighted'])

    print("----------------------------------------")
    f.write("\n----------------------------------------\n")
    print("\n4) Perceptron\n")
    f.write("\n4) Perceptron\n")
    pct = Perceptron()
    pct.fit(X_train, y_train)
    y_pred_pct = pct.predict(X_test)
    temp = print_report_perceptron(y_test, y_pred_pct)
    Per_results['accuracy-score-sum'].append(temp['accuracy-score'])
    Per_results['f1-score-macro-sum'].append(temp['f1-score-macro'])
    Per_results['f1-score-weighted-sum'].append(temp['f1-score-weighted'])

    print("----------------------------------------")
    f.write("\n----------------------------------------\n")
    print("\n5) Base-MLP Perceptron\n")
    f.write("\n5) Base-MLP Perceptron\n")
    #Iterations are deafulted to 100. I tried 5000 and it was enough to
    #covnerge to minimum
    MLPpct = MLPClassifier(hidden_layer_sizes=(100), activation="logistic", solver="sgd")
    MLPpct.fit(X_train, y_train)
    y_pred_MLPpct = MLPpct.predict(X_test)
    temp = print_report_perceptron(y_test, y_pred_MLPpct)
    Base_MLP_results['accuracy-score-sum'].append(temp['accuracy-score'])
    Base_MLP_results['f1-score-macro-sum'].append(temp['f1-score-macro'])
    Base_MLP_results['f1-score-weighted-sum'].append(temp['f1-score-weighted'])

    print("----------------------------------------")
    f.write("\n----------------------------------------\n")
    print("\n6) Top-MLP Perceptron\n")
    f.write("\n6) Top-MLP Perceptron\n")
    parameters = {'activation':['identity', 'logistic', 'tanh', 'relu'],'hidden_layer_sizes':[(30,50), (10,10,10)],'solver':['adam', 'sgd']}
    TOPMLPgsc = GridSearchCV(MLPClassifier(), parameters)
    y_pred_TOPMLP = TOPMLPgsc.fit(X_train, y_train).predict(X_test)
    temp = print_report_perceptron(y_test, y_pred_TOPMLP)
    Top_MLP_results['accuracy-score-sum'].append(temp['accuracy-score'])
    Top_MLP_results['f1-score-macro-sum'].append(temp['f1-score-macro'])
    Top_MLP_results['f1-score-weighted-sum'].append(temp['f1-score-weighted'])

def print_reports(y_test, y_pred):
    print('\nConfusion Matrix')
    print(confusion_matrix(y_test, y_pred))
    print('\nClassification Report: ')
    print(classification_report(y_test, y_pred))
    print('\nAccuracy Score: ')
    print(accuracy_score(y_test, y_pred))
    print('\nF1 score, macro: ')
    print(f1_score(y_test, y_pred, average='macro'))
    print('\nF1 score, weighted: ')
    print(f1_score(y_test, y_pred, average='weighted'))

    f.write('\nConfusion Matrix\n')
    f.write(str(confusion_matrix(y_test, y_pred)))
    f.write('\nClassification Report: \n')
    f.write(str(classification_report(y_test, y_pred)))
    f.write('\nAccuracy Score: \n')
    f.write(str(accuracy_score(y_test, y_pred)))
    f.write('\nF1 score, macro: \n')
    f.write(str(f1_score(y_test, y_pred, average='macro')))
    f.write('\nF1 score, weighted: \n')
    f.write(str(f1_score(y_test, y_pred, average='weighted')))
    return {'accuracy-score': accuracy_score(y_test, y_pred), 'f1-score-macro': f1_score(y_test, y_pred, average='macro'), 'f1-score-weighted': f1_score(y_test, y_pred, average='weighted')}

def print_report_perceptron(y_test, y_pred):
    print('\nConfusion Matrix')
    print(confusion_matrix(y_test, y_pred))
    print('\nClassification Report: ')
    print(classification_report(y_test, y_pred, zero_division=0))
    print('\nAccuracy Score: ')
    print(accuracy_score(y_test, y_pred))
    print('\nF1 score, macro: ')
    print(f1_score(y_test, y_pred, average='macro'))
    print('\nF1 score, weighted: ')
    print(f1_score(y_test, y_pred, average='weighted',zero_division=0))

    f.write('\nConfusion Matrix\n')
    f.write(str(confusion_matrix(y_test, y_pred)))
    f.write('\nClassification Report: \n')
    f.write(str(classification_report(y_test, y_pred, zero_division=0)))
    f.write('\nAccuracy Score: \n')
    f.write(str(accuracy_score(y_test, y_pred)))
    f.write('\nF1 score, macro: \n')
    f.write(str(f1_score(y_test, y_pred, average='macro')))
    f.write('\nF1 score, weighted: \n')
    f.write(str(f1_score(y_test, y_pred, average='weighted', zero_division=0)))
    return {'accuracy-score': accuracy_score(y_test, y_pred), 'f1-score-macro': f1_score(y_test, y_pred, average='macro'), 'f1-score-weighted': f1_score(y_test, y_pred, average='weighted')}


#textfile_generator()
f = open(drug_performance_file_name, "w") 
for i in range(10): 
    taskTwo()

f.write('\n\n-----------------------------------------------FINAL RESULTS-----------------------------------------------')
f.write('\n1. Average Accuracies')
f.write('\n\ta) Gaussian Naive Bayes Classifier Average Accuracy: ')
f.write('\n\t\t')
f.write(str(np.mean(NB_results['accuracy-score-sum'])))
f.write('\n\tb) Base Decision Tree Average Accuracy: ')
f.write('\n\t\t')
f.write(str(np.mean(Base_DT_results['accuracy-score-sum'])))
f.write('\n\tc) Top Decision Tree Average Accuracy: ')
f.write('\n\t\t')
f.write(str(np.mean(Top_DT_results['accuracy-score-sum'])))
f.write('\n\td) Perceptron Average Accuracy: ')
f.write('\n\t\t')
f.write(str(np.mean(Per_results['accuracy-score-sum'])))
f.write('\n\te) Base Multi-layered Perceptron Average Accuracy: ')
f.write('\n\t\t')
f.write(str(np.mean(Base_MLP_results['accuracy-score-sum'])))
f.write('\n\tf) Top Multi-layered Perceptron Average Accuracy: ')
f.write('\n\t\t')
f.write(str(np.mean(Top_MLP_results['accuracy-score-sum'])))
f.write('\n1.1 Total Average Accuracy: ')
f.write('\n\t\t')
f.write(str(np.mean(NB_results['accuracy-score-sum'] + Base_DT_results['accuracy-score-sum'] + Top_DT_results['accuracy-score-sum'] + Base_MLP_results['accuracy-score-sum'] + Top_MLP_results['accuracy-score-sum'] + Per_results['accuracy-score-sum'])))

f.write('\n\n\n2. Average Macro-Average F1')
f.write('\n\ta) Gaussian Naive Bayes Classifier Average Macro-Average F1: ')
f.write('\n\t\t')
f.write(str(np.mean(NB_results['f1-score-macro-sum'])))
f.write('\n\tb) Base Decision Tree Average Macro-Average F1: ')
f.write('\n\t\t')
f.write(str(np.mean(Base_DT_results['f1-score-macro-sum'])))
f.write('\n\tc) Top Decision Tree Average Macro-Average F1: ')
f.write('\n\t\t')
f.write(str(np.mean(Top_DT_results['f1-score-macro-sum'])))
f.write('\n\td) Perceptron Average Macro-Average F1: ')
f.write('\n\t\t')
f.write(str(np.mean(Per_results['f1-score-macro-sum'])))
f.write('\n\te) Base Multi-layered Perceptron Average Macro-Average F1: ')
f.write('\n\t\t')
f.write(str(np.mean(Base_MLP_results['f1-score-macro-sum'])))
f.write('\n\tf) Top Multi-layered Perceptron Average Macro-Average F1: ')
f.write('\n\t\t')
f.write(str(np.mean(Top_MLP_results['f1-score-macro-sum'])))
f.write('\n2.1 Total Average Macro-Average F1: ')
f.write('\n\t\t')
f.write(str(np.mean(NB_results['f1-score-macro-sum'] + Base_DT_results['f1-score-macro-sum'] + Top_DT_results['f1-score-macro-sum'] + Base_MLP_results['f1-score-macro-sum'] + Top_MLP_results['f1-score-macro-sum'] + Per_results['f1-score-macro-sum'])))

f.write('\n\n\n3. Average Weighted-Average F1')
f.write('\n\ta) Gaussian Naive Bayes Classifier Average Weighted-Average F1: ')
f.write('\n\t\t')
f.write(str(np.mean(NB_results['f1-score-weighted-sum'])))
f.write('\n\tb) Base Decision Tree Average Weighted-Average F1: ')
f.write('\n\t\t')
f.write(str(np.mean(Base_DT_results['f1-score-weighted-sum'])))
f.write('\n\tc) Top Decision Tree Average Weighted-Average F1: ')
f.write('\n\t\t')
f.write(str(np.mean(Top_DT_results['f1-score-weighted-sum'])))
f.write('\n\td) Perceptron Average Weighted-Average F1: ')
f.write('\n\t\t')
f.write(str(np.mean(Per_results['f1-score-weighted-sum'])))
f.write('\n\te) Base Multi-layered Perceptron Average Weighted-Average F1: ')
f.write('\n\t\t')
f.write(str(np.mean(Base_MLP_results['f1-score-weighted-sum'])))
f.write('\n\tf) Top Multi-layered Perceptron Average Weighted-Average F1: ')
f.write('\n\t\t')
f.write(str(np.mean(Top_MLP_results['f1-score-weighted-sum'])))
f.write('\n3.1 Total Average Weighted-Average F1: ')
f.write('\n\t\t')
f.write(str(np.mean(NB_results['f1-score-weighted-sum'] + Base_DT_results['f1-score-weighted-sum'] + Top_DT_results['f1-score-weighted-sum'] + Base_MLP_results['f1-score-weighted-sum'] + Top_MLP_results['f1-score-weighted-sum'] + Per_results['f1-score-weighted-sum'])))

f.write('\n\n\n4. Accuracy Standard Deviation')
f.write('\n\ta) Gaussian Naive Bayes Classifier Accuracy Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(NB_results['accuracy-score-sum'])))
f.write('\n\tb) Base Decision Tree Accuracy Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(Base_DT_results['accuracy-score-sum'])))
f.write('\n\tc) Top Decision Tree Accuracy Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(Top_DT_results['accuracy-score-sum'])))
f.write('\n\td) Perceptron Accuracy Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(Per_results['accuracy-score-sum'])))
f.write('\n\te) Base Multi-layered Perceptron Accuracy Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(Base_MLP_results['accuracy-score-sum'])))
f.write('\n\tf) Top Multi-layered Perceptron Accuracy Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(Top_MLP_results['accuracy-score-sum'])))
f.write('\n4.1 Total Accuracy Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(NB_results['accuracy-score-sum'] + Base_DT_results['accuracy-score-sum'] + Top_DT_results['accuracy-score-sum'] + Base_MLP_results['accuracy-score-sum'] + Top_MLP_results['accuracy-score-sum'] + Per_results['accuracy-score-sum'])))

f.write('\n\n\n5. Macro-Average F1 Standard Deviation')
f.write('\n\ta) Gaussian Naive Bayes Classifier Macro-Average F1 Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(NB_results['f1-score-macro-sum'])))
f.write('\n\tb) Base Decision Tree Macro-Average F1 Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(Base_DT_results['f1-score-macro-sum'])))
f.write('\n\tc) Top Decision Tree Macro-Average F1 Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(Top_DT_results['f1-score-macro-sum'])))
f.write('\n\td) Perceptron Macro-Average F1 Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(Per_results['f1-score-macro-sum'])))
f.write('\n\te) Base Multi-layered Perceptron Macro-Average F1 Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(Base_MLP_results['f1-score-macro-sum'])))
f.write('\n\tf) Top Multi-layered Perceptron Macro-Average F1 Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(Top_MLP_results['f1-score-macro-sum'])))
f.write('\n5.1 Total Macro-Average F1 Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(NB_results['f1-score-macro-sum'] + Base_DT_results['f1-score-macro-sum'] + Top_DT_results['f1-score-macro-sum'] + Base_MLP_results['f1-score-macro-sum'] + Top_MLP_results['f1-score-macro-sum'] + Per_results['f1-score-macro-sum'])))

f.write('\n\n\n6. Weighted-Average F1 Standard Deviation')
f.write('\n\ta) Gaussian Naive Bayes Classifier Weighted-Average F1 Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(NB_results['f1-score-weighted-sum'])))
f.write('\n\tb) Base Decision Tree Weighted-Average F1 Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(Base_DT_results['f1-score-weighted-sum'])))
f.write('\n\tc) Top Decision Tree Weighted-Average F1 Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(Top_DT_results['f1-score-weighted-sum'])))
f.write('\n\td) Perceptron Weighted-Average F1 Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(Per_results['f1-score-weighted-sum'])))
f.write('\n\te) Base Multi-layered Perceptron Weighted-Average F1 Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(Base_MLP_results['f1-score-weighted-sum'])))
f.write('\n\tf) Top Multi-layered Perceptron Weighted-Average F1 Standard Deviation: ')
f.write('\n\t\t')
f.write(str(np.std(Top_MLP_results['f1-score-weighted-sum'])))
f.write('\n6.1 Total Macro-Average Weighted-Average F1: ')
f.write('\n\t\t')
f.write(str(np.std(NB_results['f1-score-weighted-sum'] + Base_DT_results['f1-score-weighted-sum'] + Top_DT_results['f1-score-weighted-sum'] + Base_MLP_results['f1-score-weighted-sum'] + Top_MLP_results['f1-score-weighted-sum'] + Per_results['f1-score-weighted-sum'])))