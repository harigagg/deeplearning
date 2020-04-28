# Artificial Neural Network
##################################################
## ver1
##################################################
## Author: {Harika Gaggara}
## Copyright: Copyright {2020}, {NN classification}
##################################################
# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[1:, 3:13].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

LabelEncoder_X_1 = LabelEncoder()
LabelEncoder_X_2 = LabelEncoder()
X[:,1] = LabelEncoder_X_1.fit_transform(X[:,1])
X[:,2] = LabelEncoder_X_2.fit_transform(X[:,2]) 
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [1])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

X = np.array(ct.fit_transform(X), dtype=np.float)

X = X[:, 1:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 Building the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units =6, kernel_initializer = 'uniform', activation = 'relu', input_dim =11))

classifier.add(Dense(units =6, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units =1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size =10, epochs=100)

y_pred = classifier.predict(X_test)

y_pred = (y_pred>0.5)

#to predict on a single customer
prediction = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3, 60000,2,1,1,50000]])))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)


