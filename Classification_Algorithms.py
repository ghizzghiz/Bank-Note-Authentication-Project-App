# Loading Data and Overview
# Install all needed libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

# Read data file
df = pd.read_csv('data_banknote_authentication.txt', sep=',', header=None,
                          names=['variance', # variance of Wavelet Transformed image (continuous)
                                 'skewness', # skewness of Wavelet Transformed image (continuous)
                                 'kurtosis', # kurtosis of Wavelet Transformed image (continuous)
                                 'entropy', # entropy of image (continuous)
                                 'class']) # class (integer)
df.head()

df.info()

df.describe()

df.duplicated().sum() # 24 duplicate rows

"""
The dataset contains 5 attributes, 1372 observations, and no null-values. An inspection of the content of the dataset also leads to finding no issues with the dataset. The next step would be to generate any extra columns as needed.
"""

"""
# Data Pre-processing
In this step, Standardizing dataset will allow for the transformation of attributes to having a mean of 0 and a standard deviation of 1.
"""

df.values
df.values.dtype
df.values.shape
df.values.size

# Inputs (Independent variables)
X = df.values[:,:4] # 4 variables: 'variance', 'skewness', 'kurtosis', and 'entropy'
# Output (Dependent variable)
Y = df.values[:,4] # the 'class' (integer - binary 0, 1)

X_means = df.values[:,:4].mean()
X_stds = df.values[:,:4].std()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
rescaledX = scaler.fit_transform(X)
rescaledX

scaler = StandardScaler()
scaler.fit(pd.DataFrame(X))
X = pd.DataFrame(scaler.transform(X), columns = df.columns[0:4])

print(X.mean()) # either 0 or values very close to 0
print(X.std()) # around 1

from sklearn.model_selection import train_test_split
# validation size conventionally being set to 20%
validation_size = 0.2
seed = 7
Xtrain, Xvalidation, Ytrain, Yvalidation = train_test_split(X, Y, test_size = validation_size, random_state=seed)
print(Y)
print(Xtrain.shape)
print(Xvalidation.shape)

"""# **5. Logistic Regression**"""

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
models = []
models.append(('LR', LogisticRegression()))
models

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
results = []
names = []
for name, model in models:
  kfold = KFold(n_splits = 10, random_state = seed, shuffle=True)
  cv_results = cross_val_score(model, Xtrain, Ytrain, cv=kfold, scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  print("%s = %f (%f)" % (name, cv_results.mean(), cv_results.std()))

import matplotlib.pyplot as plt
plt.boxplot(results, labels = names)
plt.show()

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

LogiReg = LogisticRegression()
LogiReg.fit(Xtrain, Ytrain)
predictions = LogiReg.predict(Xvalidation)

print(accuracy_score(Yvalidation,predictions))
print(confusion_matrix(Yvalidation,predictions))
print(classification_report(Yvalidation,predictions))

print(Yvalidation)
print(predictions)

# Finalizing the Logistic regression model
from pickle import dump
# Since the Logistic regression leads to a high accuracy, we can change the test size from 20% to a greater value, say 30%.
# test size is now changed from the validation size of 20% to be set to 30%
validation_size = 0.3
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = validation_size, random_state=seed)

model = LogisticRegression()
model.fit(Xtrain, Ytrain)

dump(model, open("finalized_LogiReg.sav", "wb"))

from pickle import load
loaded_model = load(open("finalized_LogiReg.sav", "rb"))
result = loaded_model.score(Xtest, Ytest)
print('Accuracy = ', result*100) # A SLIGHT INCREASE IN THE ACCURACY

"""# **Random Forrest Algorithm**
"""

validation_size_RFA = 0.3
seed_RFA = 7
XD_Train, XD_Test, YD_Train, YD_Test = train_test_split(X, Y, test_size = validation_size_RFA, random_state=seed_RFA)
print(Y)
print(XD_Train.shape)
print(XD_Test.shape)

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

rf_model = rfc(n_estimators=200, random_state=0)

rf_model.fit(XD_Train, YD_Train)
predicted_labels = rf_model.predict(XD_Test)
rf_model.estimators_
len(rf_model.estimators_)
DX = list(df.columns.values)
DX
DX[0:4]

from sklearn import tree

fn=DX[0:4]
#cn = list(X[4])

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf_model.estimators_[100],
               feature_names = fn,
               class_names=["Genuine", "Fake"],
               filled = True);
#fig.savefig('rf_individualtree.png')

print(classification_report(YD_Test, predicted_labels))

#XD_Train, XD_Test, YD_Train, YD_Test
print(confusion_matrix(YD_Test, predicted_labels))

print('Training Accuracy : ',
      accuracy_score(YD_Train,rf_model.predict(XD_Train))*100)
print('Validation Accuracy : ',
      accuracy_score(YD_Test,rf_model.predict(XD_Test))*100)
      
dump(rf_model, open("finalized_rf.sav", "wb"))

"""# **SVM**
"""
import pandas as pd
from sklearn.svm import SVC

# Inputs (Independent variables)
X = df.values[:,:4] # 4 variables: 'variance', 'skewness', 'kurtosis', and 'entropy'
# Output (Dependent variable)
Y = df.values[:,4] # the 'class' (integer - binary 0, 1)

scaler = StandardScaler()
scaler.fit(pd.DataFrame(X))
X = pd.DataFrame(scaler.transform(X), columns = df.columns[0:4])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123)

# Create the SVM model
model = SVC()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print('Accuracy:', accuracy)

print(accuracy_score(y_test ,y_pred))
print(confusion_matrix(y_test ,y_pred))
print(classification_report(y_test ,y_pred))

dump(rf_model, open("finalized_svm.sav", "wb"))
