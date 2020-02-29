import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

test_data = pd.read_csv('../input/titanic/test.csv')

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch", "Age"]
X_train = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(X_train)

X_train_imp = imp.transform(X_train)
model = RandomForestClassifier(n_estimators=1000, max_depth=32, random_state=1)
model.fit(X_train_imp, y)

X_test_imp = imp.transform(X_test)
predictions = model.predict(X_test_imp)

output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
output.to_csv("my_submission.csv", index=False)
print("Your submission has been successfully saved!")
