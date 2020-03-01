# version 3

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')

from sklearn.ensemble import RandomForestClassifier

test_data = pd.read_csv('../input/titanic/test.csv')

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_imp = X.dropna()

X_test = pd.get_dummies(test_data[features])
X_test_imp = X_test.dropna()

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_imp, y)
predictions = model.predict(X_test_imp)

output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
output.to_csv("my_submission.csv", index=False)
print("Your submission has been successfully saved!")
