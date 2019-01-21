import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing

train_data = pd.read_csv("train.csv")
pd.S
# fig = plt.figure()
# fig.set(alpha=0.2)
# plt.subplot2grid((2, 3), (0, 0))
# a = train_data.Survived.value_counts().plot(kind="bar")
# plt.show()
# print(type(a),a)
# print(train_data.head(5))
# print(train_data.loc["Cabin"])
age_df = train_data[["Age", "Fare", "Parch", "Pclass", "SibSp"]]  # [[]] format a new dataframe colum
print(age_df.head(5))
know_age = age_df[age_df.Age.notnull()].values  # .valuse = .as_matrix() transfer into 2-d ndarray
unknow_age = age_df[age_df.Age.isnull()].values
y = know_age[:, 0]
x = know_age[:, 1:]
rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
rfr.fit(x, y)

predictAge = rfr.predict(unknow_age[:, 1:])
#print(predictAge)
train_data.loc[(train_data.Age.isnull()),"Age"] = predictAge
a = train_data.loc[(train_data.Age.notnull()),"Age"]  # loc [row,column] [column]=dataframe column=series

# one hot encoding
dummies = pd.get_dummies(train_data["Sex"],prefix="sex")
#print(dummies.head(5))
df = pd.concat([train_data,dummies],axis=1)  # axis =1 means left+right

# print(train_data.info())
scaler = sklearn.preprocessing.StandardScaler()
# age_scaler_param = scaler.fit(df[["Age"]])
# print(age_scaler_param)
aa = scaler.fit_transform(df[["Age"]])
print(aa)
