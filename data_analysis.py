import re
from string import digits

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

train_data = pd.read_csv("train.csv")
print(train_data.info())

# sns.distplot(train_data[train_data.Age.notnull()]["Age"])
# print(train_data[train_data.Age.notnull()]["Age"])
# train_data[train_data.Age.notnull()]["Age"].value_counts().plot.pie()
# plt.show()
print(train_data[train_data["Embarked"].isnull()])
train_data.loc[train_data["Embarked"].isnull(), "Embarked"] = train_data.Embarked.dropna().mode().values[0]
train_data.loc[(train_data["Cabin"].isnull()), "Cabin"] = "no_cabin"

fig = plt.figure(figsize=(10,10))
plt.subplot2grid((2, 2), (0, 0))
sns.heatmap(train_data.corr(), vmin=-1, vmax=1, annot=True, square=True)
# sns.lmplot(data=train_data,x=train_data["Cabin"],y=train_data["Survived"])
# print(train_data[["Survived",'Cabin']].groupby(["Cabin"]).mean())
# train_data.loc[train_data['Cabin'] != "seat",'Cabin'] = "have_cabin"


train_data["Cabin"] = train_data['Cabin'].str.strip(digits)
a = train_data.loc[train_data['Cabin'].str.contains(" "), "Cabin"].values
b = [i.split(" ")[-1] for i in a]
train_data.loc[train_data['Cabin'].str.contains(" "), "Cabin"] = b
print(train_data["Cabin"])

plt.subplot2grid((2, 2), (0, 1))
train_data.groupby(['Cabin'])['Survived'].mean().plot.bar()
plt.subplot2grid((2, 2), (1, 0))
train_data.groupby(["Age"])['Survived'].mean().plot.bar()

train_data['CabinLetter'] = train_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
plt.show()
