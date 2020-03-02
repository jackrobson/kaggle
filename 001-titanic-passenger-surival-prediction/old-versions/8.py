# Version 8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')

def predict(row):
    # males perish
    if row["Sex"] == "male":
        return 0
    # females aged < 1 year old survive
    if row["Age"] < 1:
        return 1
    # lower class females who spent > $20 on fare perish
    if row["Pclass"] == 3 and row["Fare"] > 20:
        return 0
    return 1

predictions = [0] * test_data.shape[0]

for index, row in test_data.iterrows():
    predictions[index] = predict(row)

output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
output.to_csv("my_submission.csv", index=False)
print("Your submission has been successfully saved!")
