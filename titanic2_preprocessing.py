import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load the training and test datasets
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# Save the 'Survived' column from the training data
survivalRate = train['Survived']
train.drop('Survived', axis=1, inplace=True)

# Function to extract titles from names
def extract_title(name):
    title_search = re.search(r"\b[A-Za-z]+\.\s?", name)
    if title_search:
        return title_search.group(0).strip().rstrip('.')  # Extract and clean the title
    return ""

# Function to group titles
def groupTitles(x):
    title = extract_title(x['Name']) 


    mr_titles = ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']
    mrs_titles = ['Countess', 'Mme']
    miss_titles = ['Mlle', 'Ms']

    if title in mr_titles:
        return 'Mr'
    elif title in mrs_titles:
        return 'Mrs'
    elif title in miss_titles:
        return 'Miss'
    elif title == 'Dr':
        if x['sex_male'] == True:  
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title  


def preprocess_data(df):

    encoded_sex = pd.get_dummies(df['Sex'], prefix='sex')
    df = pd.concat([df, encoded_sex], axis=1)
    df.drop('Sex', axis=1, inplace=True)

    encoded_embarked = pd.get_dummies(df['Embarked'], prefix='embarked')
    df = pd.concat([df, encoded_embarked], axis=1)
    df.drop('Embarked', axis=1, inplace=True)


    df['Grouped_Title'] = df.apply(groupTitles, axis=1)


    encoded_title = pd.get_dummies(df['Grouped_Title'], prefix='title')
    df = pd.concat([df, encoded_title], axis=1)
    df.drop('Grouped_Title', axis=1, inplace=True)


    age_imputer = SimpleImputer()
    age_imputed = age_imputer.fit_transform(df[['Age']])
    df['Age'] = age_imputed
    df['Age'] = df['Age'].astype(int)


    df['log_fare'] = np.log1p(df['Fare'])
    df.drop('Fare', axis=1, inplace=True)

    # Drop unnecessary columns
    df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

    return df


train = preprocess_data(train)
test = preprocess_data(test)
