"""
WORK IN PROGRESS

Uses classifier and regression analysis to predict future stock price

Need to install:
pip3 install numpy
pip3 install scipy
pip3 install scikit-learn
pip3 install matplotlib
pip3 install quandl

Features are current values and the label shall be the price.

Note !! if we look out further and set line 50 forecast_out = int(math.ceil(0.01 * len(df)))
to say 0.1 of df we get a different classifier and much lower accuracy

"""

# imports
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression



# get data
ticker = input("What is the stock ticker (e.g. GOOGL)? ")
pass_to_quandl = "WIKI/" + ticker.upper()
df = quandl.get(pass_to_quandl)

# select features
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# create new dataframe
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print()
print("First 5 days in data set")
print(df.head())
print()
print("Recent 5 days in data set")
print(df.tail())
print()

# forecast
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)


# define features with capital X and labels with lowercase y
X = np.array(df.drop(['label'], 1))
# We define X as Features for entire dataframe EXCEPT for label column, converted to a numpy array

# apply preprocessing scale to speed up learning
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# drop NAN data
df.dropna(inplace=True)

y = np.array(df['label'])

# cross_validation is to be deprecated. when it is use model_selection in line commented out
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Find best classifier from options
def best_classifier(X_train, X_test, y_train, y_test):
    # add other candidates for the classifier algorithm

    # "a" - define  "classifier" -- Support Vector Regression from Scikit-Learn's svm package
    a = svm.SVR()
    # With Scikit-Learn (sklearn) you train with .fit
    a.fit(X_train, y_train)
    # measure & print accuracy
    a_score = a.score(X_test, y_test)

    # 'b' - define  "classifier" -- LinearRegression(kernel='linear')
    # - n_jobs invokes threading
    b = LinearRegression(n_jobs=-1)
    b.fit(X_train, y_train)
    b_score = b.score(X_test, y_test)

    # 'c' - define  "classifier" -- LinearRegression(kernel='poly')
    c = LinearRegression(n_jobs=-1)
    c.fit(X_train, y_train)
    c_score = c.score(X_test, y_test)

    # 'd' - define  "classifier" -- LinearRegression(kernel='rbf')
    d = LinearRegression(n_jobs=-1)
    d.fit(X_train, y_train)
    d_score = d.score(X_test, y_test)

    # 'e' - define  "classifier" -- LinearRegression(kernel='sigmoid')
    e = LinearRegression(n_jobs=-1)
    e.fit(X_train, y_train)
    e_score = e.score(X_test, y_test)

    best_score = max(a_score, b_score, c_score, d_score, e_score)
    if best_score == a_score:
        best_method = "svm.SVR()"

    elif best_score == b_score:
          best_method = "LinearRegression(kernel='linear')"

    elif best_score == c_score:
          best_method = "LinearRegression(kernel='poly')"

    elif best_score == d_score:
          best_method = "LinearRegression(kernel='rbf')"

    elif best_score == e_score:
          best_method = "LinearRegression(kernel='rbf')"

    else:
          best_method = "No winner"

    print("The best classifier is:", best_method, "Winning accuracy is:", best_score)
    print("Beware, when we changed the sample size in line 50 SVM won but accuracy dropped to 60%")
    print()
    return  best_method, best_score

my_output = best_classifier(X_train, X_test, y_train, y_test)
