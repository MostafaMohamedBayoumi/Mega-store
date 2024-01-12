import pickle
import pandas as pd
import seaborn as sns
import preprocessing as pre
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


data = pd.read_csv("megastore-regression-dataset.csv")
Y = data.loc[:, 'Profit']
X = data.drop('Profit', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=42)

X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

X_train = pre.pre_processing(X_train)
X_train_num, X_train_cat = pre.numerical_Categorical(X_train)
for col in X_train_cat:
    encoder = ce.OrdinalEncoder(handle_unknown='value', handle_missing='value')  # SubCategory
    X_train_cat.loc[:, col] = encoder.fit_transform(X_train_cat.loc[:, col])
    pickle.dump(encoder, open('encoders/' + col + '.sav', 'wb'))

# feature selection
fs_train = SelectKBest(score_func=f_regression, k=7)
feature_score = pd.concat([pd.DataFrame(X_train_cat.columns),
                           pd.DataFrame(fs_train.fit(X_train_cat, y_train).scores_)],
                          axis=1)
feature_score.columns = ['Categorical Features', 'F_Score']
feature_score = feature_score.nlargest(7, columns='F_Score')

L_train = []
for col in X_train_cat.columns:
    if col not in feature_score['Categorical Features'].values:
        L_train.append(col)

X_train_cat.drop(L_train, axis=1, inplace=True)
X_train_num.loc[:, 'Profit'] = y_train
# visualization of the correlation between features and target
plt.subplots(figsize=(12, 8))
corr = X_train_num.corr()
sns.heatmap(corr, annot=True)
plt.show()

top_features = corr.index[abs(corr.loc[:, 'Profit']) > 0.2]
X_train_num = X_train_num[top_features].drop(columns=['Profit'])
X_train = pd.concat([X_train_num, X_train_cat], axis=1)
pickle.dump(X_train.loc[0:1, :], open('models/features.pkl', 'wb'))
# end feature selection

# transforms the existing features to higher degree features.
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)
pickle.dump(poly_features, open("models/Polynomial.sav", "wb"))

# fit the transformed features to Linear Regression
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
pickle.dump(poly_model, open("models/poly_model.sav", "wb"))

# fit on other regression models to find the best model fit in our task
X_train.describe(include="all")

rid = Ridge()
rid.fit(X_train, y_train)
pickle.dump(poly_model, open("models/ridge_model.sav", "wb"))

lass = Lasso()
lass.fit(X_train, y_train)
pickle.dump(poly_model, open("models/Lasso_model.sav", "wb"))

elasticNet = ElasticNet()
elasticNet.fit(X_train, y_train)
pickle.dump(poly_model, open("models/elasticNet_model.sav", "wb"))

model = LinearRegression()
model.fit(X_train, y_train)
pickle.dump(model, open("models/multivariable.sav", "wb"))

print("training done")

# testing
X_test = pre.pre_processing(X_test)
X_test_num, X_test_cat = pre.numerical_Categorical(X_test)
for col in X_test_cat:
    encoder = pickle.load(open('encoders/' + col + '.sav', 'rb'))
    X_test_cat.loc[:, col] = encoder.transform(X_test_cat.loc[:, col])

X_test_cat.drop(L_train, axis=1, inplace=True)
X_test_num = X_test_num[top_features[:-1]]
X_test = pd.concat([X_test_num, X_test_cat], axis=1)

print("elasticNet score : ", elasticNet.score(X_test, y_test))

print("ridge model score : ", rid.score(X_test, y_test))

print("lasso model score : ", lass.score(X_test, y_test))

print("Polynomial model Score : ", poly_model.score(poly_features.fit_transform(X_test), y_test))

print("Linear-multivariable model score  : ", model.score(X_test, y_test))

#######################################################################################################
