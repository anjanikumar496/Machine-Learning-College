

import pandas as pd
import numpy as np

# import plotting libraries
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set(font_scale=1.5)

# import libraries for model validation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# import libraries for metrics and reporting
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import classification_report

from collections import Counter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold



class DataPreprocessin():
    def Process():
        cleveland=pd.read_excel('cleveland.xlsx')
        statlog= pd.read_excel('Statlog.xlt')

        print(cleveland.shape)
        print(statlog.shape)

        cleveland.isnull().sum()
        statlog.isnull().sum()
        #  As we can see there are no null value present in the training as wel as in the testing data set¶

        plt.figure(figsize=(16, 7))
        g = sns.heatmap(cleveland[
                            ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                             'slope', 'ca', 'thal', 'target']].corr(), annot=True, cmap='coolwarm')



        # Detecting the Outlair¶

        # Outlier Detection using Tuckey's Method

        def detect_outlier(df, n, features):
            outlier_indecies = []
            for col in features:
                Q1 = np.percentile(df[col], 25)  # Finding the Quartile Range
                Q3 = np.percentile(df[col], 75)  # Finding the 3rd Quartile Range

                IQR = Q3 - Q1

                # Setting the outlier Steps
                outlier_steps = 1.5 * IQR

                # findinfg the outlier indices

                outlier_indices_col = df[(df[col] < Q1 - outlier_steps) | (df[col] > Q3 + outlier_steps)].index
                outlier_indecies.extend(outlier_indices_col)

            outlier_indecies = Counter(outlier_indecies)

            multiple_outlier = list(k for k, v in outlier_indecies.items() if v > n)
            return multiple_outlier

        outlier_to_drop = detect_outlier(cleveland, 2,
                                         ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                                          'oldpeak', 'slope', 'ca', 'thal', 'target'])
        print(outlier_to_drop)

        # so with the above observation I can see there is no outlier.

        ## Row 87 having the Outlair so we have to remove
        statlog = statlog.drop(outlier_to_drop, axis=0)

        print(cleveland)

        print(cleveland.dtypes)

        print(statlog.dtypes)

        print(cleveland['sex'].value_counts())

        g = sns.factorplot(x='sex', y='target', data=cleveland, kind='bar', palette='muted')

        g = g.set_ylabels('target')

        ## With the above observation female achiving the target.
        cleveland.describe()

        statlog.describe()

        x = cleveland.drop('target', axis=1)
        y = cleveland['target']

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        data = {'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test}

        return data







