import re
from string import digits

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def divide_cabin(x):
    x = x.strip(digits)
    return x.split(" ")[-1] if " " in x else x


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print(train_df.info())
# ignore: PassengerId, Name,Ticket,SibSp,Parch
# total corr
plt.figure(figsize=(20, 13))
plt.subplot2grid((3, 3), (0, 0))
sns.heatmap(train_df.corr(), vmin=-1, vmax=1, annot=True)
# Cabin
plt.subplot2grid((3, 3), (0, 1))
train_df.loc[train_df["Cabin"].isnull(), "Cabin"] = "seat"
# train_df['Cabin'] = train_df['Cabin'].apply(divide_cabin)
train_df['Cabin'] = train_df['Cabin'].apply(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
print(train_df["Cabin"])
train_df.groupby("Cabin")["Survived"].mean().plot.bar()
# Pclass
plt.subplot2grid((3, 3), (0, 2))
train_df.groupby("Pclass")["Survived"].mean().plot.bar()
# Sex
plt.subplot2grid((3, 3), (1, 0))
# train_df.groupby("Sex")["Survived"].mean().plot.bar()
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=train_df, ci=0, estimator=np.mean, palette="Set2")
# Age need to fill in (by Name/Parch...)
plt.subplot2grid((3, 3), (1, 1))
train_df["Age"] = train_df.loc[train_df["Age"].notnull(), "Age"].apply(lambda x: int(x / 10) * 10)
# train_df.groupby("Age")["Survived"].mean().plot.bar()
# sns.distplot(a=train_df.loc[train_df["Age"].notnull(),"Age"],bins=10)

# fare
plt.subplot2grid((3, 3), (1, 2))
# train_df.loc[train_df["Survived"]==1,"Fare"].plot.line()
#train_df.loc[train_df["Survived"] == 1, ["Fare", "Embarked"]].groupby("Embarked")["Fare"].plot()
sns.barplot(x="Parch",y="Survived",data=train_df,ci=0,palette="Set2")
# age & sex
plt.subplot2grid((3, 3), (2, 0))
sns.boxplot(x="Sex", y="Age", data=train_df, hue="Survived", palette="Set2", width=0.5)
plt.subplot2grid((3, 3), (2, 1))
sns.violinplot(x="Sex", y="Age", data=train_df, hue="Survived", scale="count", split=True, palette="Set2")
plt.subplot2grid((3, 3), (2, 2))
# sns.FacetGrid(data=train_df,hue="Survived").map(sns.kdeplot,"Age").add_legend()  # will gen a new pic
#train_df["Age_int"] = train_df.loc[train_df["Age"].notnull(), "Age"].astype(int)
#average_age = train_df[["Age_int", "Survived"]].groupby(['Age_int'], as_index=False).mean()
sns.barplot(x="Age", y="Survived", data=train_df, ci=0,estimator=np.mean,palette="Set2")
plt.show()
