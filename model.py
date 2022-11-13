# Importing the libraries
import pickle
import pandas as pd
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('C:\Dataset\heart.csv')

# Menyimpan nilai-nilai dalam dataset ke dalam variabel array
array = data.values

# Menyimpan data fitur-fitur pada dataset ke dalam variabel X
X = array[:,0:13]

# Menyimpan data label pada dataset ke dalam variabel Y
Y = array[:,13]

# Mendefinisikan ukuran testing data dan seed untuk random state
test_size = 0.20
seed = 7

# Memisahkan data menjadi training (dan validation) set dan testing set
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

# Mendefinisikan algoritma AdaBoost
Tree = DecisionTreeClassifier()

# Melatih data training dengan algoritma AdaBoost
Tree.fit(X_train, Y_train)

# Saving model to disk
filename = 'model_tree.pkl'
pickle.dump(Tree, open(filename, 'wb'))