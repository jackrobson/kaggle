# Version 7

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')

def predict(row):
    # assumes males perish
    if row["Sex"] == "male":
        return 0
    # assumes females aged < 1 year old survive
    if row["Age"] < 1:
        return 1
    # assumes lower class females aged 1-3 perish
    if row["Pclass"] == 3 and row["Age"] >= 1 and row["Age"] <= 3:
        return 0
    # assumes lower class females aged 4-5 survive
    if row["Pclass"] == 3 and row["Age"] >= 4 and row["Age"] <= 5:
        return 1
    # assumes lower class females aged 6-12 perish
    if row["Pclass"] == 3 and row["Age"] >= 6 and row["Age"] <= 12:
        return 0
    # assumes lower class females aged 13-18 survive
    if row["Pclass"] == 3 and row["Age"] >= 13 and row["Age"] <= 18:
        return 1
    # assumes lower class females aged 19+ perish
    if row["Pclass"] == 3 and row["Age"] >= 19:
        return 0
    return 1

predictions = [0] * test_data.shape[0]

for index, row in test_data.iterrows():
    predictions[index] = predict(row)

output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
output.to_csv("my_submission.csv", index=False)
print("Your submission has been successfully saved!")
