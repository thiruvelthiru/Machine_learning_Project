import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv("loan_dataset.csv")

#preprocessing
data.isnull().sum()
# Dropping the missing values
data = data.dropna()

data.isnull().sum()

data.head()

# Label Encoding replacing Y/N into numerical values
data.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)
data = data.replace({"Dependents":{'3+':4}})
data.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
              'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
data.head()
data.columns
'''
sns.heatmap(data.corr(),cmap="magma_r",annot=True)
plt.rcParams["figure.figsize"]=(12,8)
plt.show()
'''

X = data.drop(columns=['Loan_ID', 'Loan_Status','Gender'], axis=1)
Y = data['Loan_Status']

# Splitting data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=42)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
'''
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy of training data :', training_data_accuracy)
'''
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy of test data :', test_data_accuracy)


#Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
#Property_Area 


res=classifier.predict([[1,1,1,1,12841,10968,349,360,1,1]])
if res==1:
    print("He can get the loan")
else:
    print("He can't get the loan")


res1=classifier.predict([[1,1,1,0,300,30000,39,30,0,1]])
if res1==1:
    print("He can get the loan")
else:
    print("He can't get the loan")














