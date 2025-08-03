
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
import pickle
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('winequality-red.csv')

# Correlation analysis and visualization (optional)
# sns.countplot(x='quality', data=data)
# plt.show()

# Detect and remove outliers
def outliers(df, ft):
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df.index[(df[ft] < lower_bound) | (df[ft] > upper_bound)]

index_list = []
features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
            'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']

for feature in features:
    index_list.extend(outliers(data, feature))

def remove(df, ls):
    ls = sorted(set(ls))
    return df.drop(ls)

data_cleaned = remove(data, index_list)

# Further filtering
data_cleaned = data_cleaned.loc[(data_cleaned['fixed acidity'] < 11) &
                                (data_cleaned['volatile acidity'] < 0.95) &
                                (data_cleaned['residual sugar'] < 3.2) &
                                (data_cleaned['chlorides'] < 0.10) &
                                (data_cleaned['free sulfur dioxide'] < 35) &
                                (data_cleaned['total sulfur dioxide'] < 90) &
                                (data_cleaned['density'] < 1.000) &
                                (data_cleaned['pH'] < 3.65) &
                                (data_cleaned['sulphates'] < 0.9) &
                                (data_cleaned['alcohol'] < 13) &
                                (data_cleaned['sulphates'] > 0.4) &
                                (data_cleaned['chlorides'] > 0.05)]

# Map quality scores to binary categories
data_cleaned['quality'] = data_cleaned['quality'].map({3: 'bad', 4: 'bad', 5: 'bad', 6: 'good', 7: 'good', 8: 'good'})
le = LabelEncoder()
data_cleaned['quality'] = le.fit_transform(data_cleaned['quality'])

# Split features and target
X = data_cleaned.drop('quality', axis=1)
y = data_cleaned['quality']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate models
def evaluate(model, name):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"\n{name} Results:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Accuracy:", model.score(x_test, y_test))

# Evaluate SVM
evaluate(SVC(), "Support Vector Machine")

# Evaluate Naive Bayes
evaluate(GaussianNB(), "Naive Bayes")

# Evaluate Random Forest
modelrfc = RandomForestClassifier(n_estimators=100)
evaluate(modelrfc, "Random Forest")

# Save best model (Random Forest) to disk
with open('model.pkl', 'wb') as f:
    pickle.dump(modelrfc, f)
