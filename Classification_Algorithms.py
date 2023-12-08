
## **2. Loading Data and Overview**
# Install all needed libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

"""* The dataset contains 5 attributes, 1372 observations, and no null-values. An inspection of the content of the dataset also leads to finding no issues with the dataset. The next step would be to generate any extra columns as needed.

## **3. Data Exploration**
"""

# Box plot and outliers
for i in range(0,5):
  plt.subplot(2,3,i+1)
  #plt.boxplot(df[df.columns[i]])
  sns.boxplot(data=df, y = df[df.columns[i]])
  plt.title(df.columns[i].capitalize())
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.85, wspace=0.4, hspace=0.6)
plt.suptitle('Boxplots of dataset variables', fontsize=15, fontweight='bold')
plt.show()

# Box plot per class
plt.figure(figsize=(10,5))
for i in range(0,4):
  plt.subplot(2,3,i+1)
  sns.boxplot(data=df, x='class', y = df[df.columns[i]])
  plt.title(df.columns[i].capitalize())
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.85, wspace=0.4, hspace=0.8)
plt.suptitle('Boxplots of dataset variables', fontsize=15, fontweight='bold')
plt.show()

"""### Results from plot
* The distribution of ....
* .......
* .......
"""

# Histogram for all dataset variables
for i in range(0,5):
  plt.subplot(2,3,i+1)
  plt.hist(df[df.columns[i]])
  plt.title(df.columns[i].capitalize())
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.85, wspace=0.4, hspace=0.6)
plt.suptitle('Histogram of dataset variables', fontsize=15, fontweight='bold')
plt.show()

# Distribution for all dataset variables
for i in range(0,5):
  plt.subplot(2,3,i+1)
  sns.distplot(df[df.columns[i]])
  plt.title(df.columns[i].capitalize())
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.85, wspace=0.4, hspace=0.6)
plt.suptitle('Histogram of dataset variables', fontsize=15, fontweight='bold')
plt.show()

"""### Results from plot
* .......
* .......
* .......
"""

# Pair plot
sns.pairplot(df.loc[:, df.columns !='class'])#.fig.suptitle('Pair plot of all variables except class')
sns.set_style("whitegrid", {'grid.linestyle': '--'})

"""### Results from plot
* .......
* .......
* .......
"""

# How many datapoints per class in our data
class_0 = df[df['class']==0]['class'].count()
class_1 = df[df['class']==1]['class'].count()

classes = np.array([class_0,class_1])
classes

# Pie chart for classes in our data
labels = ['Class 0: Authentic','Class 1: Fraudulent']
plt.pie(classes, autopct='%1.1f%%')
plt.title('Percentages of classes in the dataset', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.5, 1), labels = labels, loc='best')
plt.show()

"""### Results from plot
* The pie plot for the variable class shows that the dataset contains more auhtentic bank notes data than inauthentic ones.
* ............
"""

# Exploring variables by class (as a different color in the scatter plot)
sns.pairplot(df, hue='class')
sns.set_style("whitegrid", {'grid.linestyle': '--'})

"""### Results from plot
* This plot allows for more meaningful observations where the class being 0 - authentic - (in blue) or 1 - inauthentic - (in orange) for each of the other four variables.
* Skewness and kurtosis look like they are negatively correlated even for both categories of class, only the 0 (authentic in blue) has higher skewneess values than the other category but higher kusrtosis for the latter than the class 0.
* The variance and entropy look somewhat positively correlated (the correlation is actually equal to 0.27) and the classes are also somewhat separated.
* The distribution of each variable per class is another intersting aspect of this graph. For the entropy, it looks like both classes are overlapping in terms of distribution. The distribution of kurtosis for class 0 is narrower and taller than class 1 which has a longer right tail. For skewness, the class 0 is farther to the right (has higher values than the other class) and is taller. And the same is observed for the variance.
"""

corr = df.corr()
corr

corr[abs(corr)>0.5]

# Correlation matrix
plt.figure(figsize=(6,6))
sns.heatmap(corr, cmap='mako', annot=True, vmax=1, vmin=-1)
plt.suptitle("Correlation matrix")
plt.show()

"""### Results from plot
* Only 6 strongly correlated variables (6 regions shown in the graph) and they are all negatively correlated: kurtosis and skewness, then there is class and variance, then lastly the entropy and skewness.
* However, looking back at the previous graph, entropy and skewness look like they are best represented by a polynomial of negative coefficient rather than a linear relation.

## **4. Data Pre-processing**
In this step, Standardizing dataset will allow for the transformation of attributes to having a mean of 0 and a standard deviation of 1.
"""

df.values
df.values.dtype
df.values.shape
df.values.size

#input_vals
X = df.values[:,:4] # 4 variables: 'variance', 'skewness', 'kurtosis', and 'entropy'
#output_vals
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
X # compare X with df: values have changed

df

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

"""* Looking at the confusion matrix, only ...... values were predicted wrong.

* Recall is retio of ...
.......

* Precision is the ratio ........
.......

* F-score is ........
.......

* Support .....
.......
"""

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

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

rf_model = rfc(n_estimators=200, random_state=0)

rf_model.fit(Xtrain, Ytrain)

predicted_labels = rf_model.predict(Xtest)

rf_model.estimators_

len(rf_model.estimators_)

X = list(df.columns.values)

from sklearn import tree

fn=X[0:4]
#cn = list(X[4])

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf_model.estimators_[100],
               feature_names = fn,
               class_names=["Genuine", "Fake"],
               filled = True);
#fig.savefig('rf_individualtree.png')

"""# **SVM**
"""
import pandas as pd
from sklearn.svm import SVC

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
