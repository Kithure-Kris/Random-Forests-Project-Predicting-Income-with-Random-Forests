def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

#Investigate the data
#https://archive.ics.uci.edu/ml/datasets/census+income
income_data = pd.read_csv('income.csv', header= 0, delimiter=',')
print(income_data.iloc[0])

#Changing column types
income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)

print(income_data['native-country'].value_counts())

income_data['country-int'] = income_data['native-country'].apply(lambda row: 0 if row == 'United States' else 1)

#Create labels and data
labels = income_data[['income']]
data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int", "country-int"]]

#Split data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

#Create the Random Forest
forest = RandomForestClassifier(random_state=1)
forest.fit(train_data, train_labels)
print(forest.score(test_data, test_labels))