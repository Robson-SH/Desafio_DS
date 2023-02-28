import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def transforming_data(train, test):
    train = train.dropna(subset=["Age"])
    test = test.dropna(subset=["Age"])
    train['Sex'] = (train['Sex'] == 'female').astype(int)
    test['Sex'] = (test['Sex'] == 'female').astype(int)
    train = train.dropna(subset=['Embarked'])
    test = test.dropna(subset=['Embarked'])
    train = train.drop(columns=['Cabin','Ticket', 'PassengerId', 'Name','SibSp','Parch'])
    test = test.drop(columns=['Cabin','Ticket', 'PassengerId', 'Name','SibSp','Parch'])
    train['Pclass'] = train['Pclass'].astype('category')
    train['Sex'] = train['Sex'].astype('category')
    train['Embarked'] = train['Embarked'].astype('category')
    test['Pclass'] = test['Pclass'].astype('category')
    test['Sex'] = test['Sex'].astype('category')
    test['Embarked'] = test['Embarked'].astype('category')
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    enc = OneHotEncoder(handle_unknown = 'ignore')
    enc_data = pd.DataFrame(enc.fit_transform(train[['Pclass','Sex','Embarked']]).toarray())
    enc_data_test = pd.DataFrame(enc.fit_transform(test[['Pclass','Sex','Embarked']]).toarray())    
    train.join(enc_data).drop(columns=['Pclass','Embarked'])
    test.join(enc_data_test).drop(columns=['Pclass','Embarked'])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train = pd.get_dummies(train, columns=['Pclass', 'Sex', 'Embarked'])
    test = pd.get_dummies(test, columns=['Pclass', 'Sex', 'Embarked'])
    return train, test

def predict(train_fixed, test_fixed):
    print(train_fixed.head(0)) 
    X_train = train_fixed
    X_train = X_train.iloc[:,1:]
    print(X_train.head(0))
    y_train = train_fixed
    y_train = y_train.iloc[:,0]
    print(y_train.head(0))
    model = xgb.XGBClassifier()
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    model.fit(X_train, y_train)
    test_fixed = test_fixed[['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_0', 'Sex_1',
         'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    y_pred = model.predict(test_fixed)
    test_fixed['pred']=y_pred
    output=test_fixed
    return output