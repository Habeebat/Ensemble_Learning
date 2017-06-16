import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.ensemble import ExtraTreesRegressor
import statsmodels.api as sm
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

#importing the dataset

dataset   = pd.read_csv("kc-house-data.csv",encoding = "ISO-8859-1")

X         = dataset[["sqft_above","sqft_basement","sqft_lot","sqft_living","floors","bedrooms",
                     "yr_built","lat","long","bathrooms"]].values
Y         = dataset["price"].values
zipcodes  = pd.get_dummies(dataset["zipcode"]).values
condition = pd.get_dummies(dataset["condition"]).values
grade     = pd.get_dummies(dataset["grade"]).values
X         = np.concatenate((X,zipcodes),axis=1)
X         = np.concatenate((X,condition),axis=1)
X         = np.concatenate((X,grade),axis=1)

#building multivariant regression stats model
model = sm.OLS(dataset["price"],X)
results = model.fit()
print(results.summary())

#building linear regression model
clf   = LinearRegression()
clf.fit(X, dataset["price"].values)
scores = cross_validation.cross_val_score(clf,X , dataset["price"].values, cv=3)
print("Linear Regression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(clf.coef_)
print("LinearRegression # coeffs :" + str(clf.coef_.shape[0]))

#lasso model
clf    = Lasso(max_iter = 100000000)
clf.fit(X, dataset["price"].values)
scores = cross_validation.cross_val_score(clf,X , dataset["price"].values, cv=5)
print("Lasso Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(clf.coef_)
print("Lasso # coeffs :" + str(clf.coef_[clf.coef_>0].shape[0]))

#extraTrees regressor
clf            = ExtraTreesRegressor()
parameters     = {'max_depth':np.arange(1,15)}
clfgrid        = grid_search.GridSearchCV(clf, parameters)
clfgrid.fit(X, dataset["price"].values)
scores = cross_validation.cross_val_score(clf,X , dataset["price"].values, cv=5)
print("Extratrees Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#######################################################################################
