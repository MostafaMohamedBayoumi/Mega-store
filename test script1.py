import numpy as np
from sklearn.model_selection import train_test_split
import preprocessing as pre
import pickle as pick
import pandas as pd

# regression
regression_data = pd.read_csv("megastore-regression-dataset.csv")
selected_features = pick.load(open('models/features.pkl', 'rb'))

print(regression_data.dtypes)
Y = regression_data.loc[:, 'Profit']
X = regression_data.drop('Profit', axis=1)

X = pre.pre_processing(X)
X = X[selected_features.columns]
X_test_num, X_test_cat = pre.numerical_Categorical(X)

statistics = pick.load(open('models/statistics1.pkl', 'rb'))
# handel Na values
skew_values = X_test_num.skew()
for column in X_test_num.columns[X_test_num.isnull().any()]:
    skewness = skew_values[column]
    if abs(skewness) < 0.5:  # Assuming a skewness threshold of 0.5
        X[column].fillna(statistics.at[column, 'mean'], inplace=True)
    else:
        X[column].fillna(statistics.at[column, 'median'], inplace=True)

for col in X_test_cat.columns:
    X[col].fillna(statistics.at[col, 'mode'])

if abs(Y.skew()) < 0.5:  # Assuming a skewness threshold of 0.5
    Y.fillna(statistics.at['Profit', 'mean'], inplace=True)
else:
    Y.fillna(statistics.at['Profit', 'median'], inplace=True)

for col in X_test_cat:
    encoder = pick.load(open('encoders/' + col + '.sav', 'rb'))
    X.loc[:, col] = encoder.transform(X.loc[:, col])



poly_features = pick.load(open("models/Polynomial.sav", "rb"))
poly_model = pick.load(open("models/poly_model.sav", "rb"))
print("Polynomial model Score : ", poly_model.score(poly_features.transform(X), Y))

random_forest_reg = pick.load(open("models/random_forest.sav", "rb"))
multivariable = pick.load(open("models/multivariable.sav", "rb"))
elasticNet_model = pick.load(open("models/elasticNet_model.sav", "rb"))

print("random_forest_reg model Score : ", random_forest_reg.score(X, Y))
print("multivariable model Score : ", multivariable.score(X, Y))
print("elasticNet  model Score : ", elasticNet_model.score(X, Y))

print("\n")
# ################################################################################################################## #
