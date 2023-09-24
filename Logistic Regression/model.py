# Logistic Regression

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
#from pickle import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)

# Feature Scaling
#StandardScaler performs the task of Standardization. Our dataset contains variable values that 
#are different in scale. For e.g. an age  20-70 and SALARY column with values on scale 10000-80000.
#As these two columns are different in scale, they are Standardized to have common scale while 
#building machine learning model.
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 

# Fitting Logistic Regression to the Training set
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#To get test Accuracy
test_acc = accuracy_score(y_test, y_pred)
print("The Accuracy for Test Set is {}".format(test_acc*100))

# Making the Confusion Matrix
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(12,6))
plt.title("Confusion Matrix")
sns.heatmap(cm, annot=True,fmt='d', cmap='Blues')
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")

#Creating a classification report for the model.
print(classification_report(y_test,y_pred))



#Test prediction
user_age_salary=[[12,300000]]
scaled_result = sc.transform(user_age_salary)
res=classifier.predict(scaled_result)
if res==1:
    print("He can buy the car")
else:
    print("He can't buy the car")








from pickle import dump
dump(classifier, open('model.pkl', 'wb'))
# save the scaler
dump(sc, open('scaler.pkl', 'wb'))

