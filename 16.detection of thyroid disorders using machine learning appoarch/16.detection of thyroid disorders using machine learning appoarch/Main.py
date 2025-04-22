from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import webbrowser
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

global filename

global X,Y
global dataset
global main
global text
accuracy = []
precision = []
recall = []
fscore = []
global X_train, X_test, y_train, y_test, predict_cls
global classifier

main = tkinter.Tk()
main.title("Detection Of Thyroid Disorders Using Machine Learning Appoarch") #designing main screen
main.geometry("1300x1200")

 
#fucntion to upload dataset
def uploadDataset():
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,"Dataset before preprocessing\n\n")
    text.insert(END,str(dataset.head()))
    text.update_idletasks()
    label = dataset.groupby('FLAG').size()
    label.plot(kind="bar")
    plt.title("Blockchain Fraud Detection Graph 0 means Normal & 1 means Fraud")
    plt.show()
    
#function to perform dataset preprocessing
def trainTest():
    global X,Y
    global dataset
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    #replace missing values with 0
    dataset.fillna(0, inplace = True)
    Y = dataset['FLAG'].ravel()
    dataset = dataset.values
    X = dataset[:,4:dataset.shape[1]-2]
    X = normalize(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X = X[0:5000]
    Y = Y[0:5000]     
    print(Y)
    print(X)
    text.insert(END,"Dataset after features normalization\n\n")
    text.insert(END,str(X)+"\n\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset: "+str(X.shape[1])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% dataset records used to train ML algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to train ML algorithms : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")

def runLogisticRegression():
    global X,Y, X_train, X_test, y_train, y_test
    global accuracy, precision,recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    text.delete('1.0', END)
    lr = LogisticRegression() 
    lr.fit(X, Y) 
    predict = lr.predict(X_test)
    calculateMetrics("Logistic Regression", predict, y_test)

def runMLP():
    mlp = MLPClassifier() 
    mlp.fit(X_train, y_train) 
    predict = mlp.predict(X_test)
    calculateMetrics("MLP", predict, y_test)
    

    

def runRF():
    global predict_cls
    rf = RandomForestClassifier() 
    rf.fit(X_train, y_train) 
    predict = rf.predict(X_test)
    predict_cls = rf
    calculateMetrics("Random Forest", predict, y_test)

def predict():
    global predict_cls
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    X = dataset[:,4:dataset.shape[1]-2]
    X1 = normalize(X)
    prediction = predict_cls.predict(X1)
    print(prediction)
    for i in range(len(prediction)):
        if prediction[i] == 0:
            text.insert(END,"Test DATA : "+str(X[i])+" ===> PREDICTED AS NORMAL\n\n")
        else:
            text.insert(END,"Test DATA : "+str(X[i])+" ===> PREDICTED AS Disease\n\n")
    
    





font = ('times', 16, 'bold')
title = Label(main, text='Detection Of Thyroid Disorders Using Machine Learning Appoarch')
title.config(bg='greenyellow', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload & Preprocess Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

traintestButton = Button(main, text="Generate Train & Test Model", command=trainTest)
traintestButton.place(x=330,y=550)
traintestButton.config(font=font1) 

lrButton = Button(main, text="Run Random  Algorithm", command=runRF)
lrButton.place(x=630,y=550)
lrButton.config(font=font1)

mlpButton = Button(main, text="Predict Disease", command=predict)
mlpButton.place(x=950,y=550)
mlpButton.config(font=font1)





main.config(bg='LightSkyBlue')
main.mainloop()
