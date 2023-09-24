from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
#Preprocessing the given data
def preprocessing(sentence):
    new_sentence = sentence[:len(sentence) - 3]
    return new_sentence.lower()
# Load the data X-statement Y-label
X = []
Y = []
filenames = ['Datasets/amazon data.txt', 'Datasets/imdb data.txt','Datasets/yelp data.txt']
for f in filenames:
    file = open(f, 'r') # r means read because you reading the data form the dataset   #r -> read w->write
    lines = file.readlines()
    for line in lines:
        sentence = preprocessing(line)
        X.append(sentence)
        Y.append(line[-2])
    file.close()
    
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=2) #612, 2
# Vectorize text data to numbers
vec = CountVectorizer(stop_words='english')
x_train = vec.fit_transform(x_train).toarray()
x_test = vec.transform(x_test).toarray()

model = MultinomialNB()
model.fit(x_train, y_train)
# Results of the model
y_pred = model.predict(x_test)

#Accuracy
from sklearn.metrics import accuracy_score
accuray = accuracy_score(y_pred, y_test)
print("Accuracy:", accuray)


import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1])

cm_display.plot()
plt.show()



#Predicti0
review="action is not good"
comment=[review]
res=vec.transform(comment).toarray()
ans=model.predict(res)
#predict_prob = model.predict_proba(res)
#print("predict_prob",predict_prob)
if ans[0]=='0':
    print('Negative')
else:
    print("Positive")







import pickle
pickle.dump(model, open("vectorizer.pickle", "wb"))

