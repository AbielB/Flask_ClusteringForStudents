# Importing the libraries
import pickle
import pandas as pd
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('hasil_cluster2.csv')
#make classification with Decision Tree
X = data.iloc[:, 0:5]
y = data.iloc[:, 5]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=7)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
# save the model to disk
filename = 'student_tree.pkl'
pickle.dump(model, open(filename, 'wb'))