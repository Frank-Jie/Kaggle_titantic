import pandas as pd
from sklearn import preprocessing

train_df = pd.read_csv("train.csv")
train_df.info()

test_df = pd.read_csv("test.csv")
test_df["Survived"] = 0
test_df.info()
test_df.drop(test_df["Fare"].isnull().index)

combin_df = train_df.append(test_df)

scaler = preprocessing.StandardScaler()
combin_df["Age"] = scaler.fit_transform(combin_df["Age"].values.reshape(-1, 1))

combin_df["Fare"] = pd.qcut(combin_df["Fare"], 5, labels=False)
combin_df[combin_df["Embarked"].isnull()] = combin_df["Embarked"].dropna().mode().values[0]

