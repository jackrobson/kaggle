# Version 6

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')

#
# After studying train.csv in a pivot Tables on Google Sheets,
# it looks like lower class girls, aged 1-3 or 6+ perish but < 12 months surive.
# This model uses this gathered information to predict.
#

def predict(row):
    if row["Sex"] == "male":
        return 0
    if row["Age"] < 1:
        return 1
    if row["Pclass"] == 3 and row["Age"] >= 1 and row["Age"] <= 3:
        return 0
    if row["Pclass"] == 3 and row["Age"] >= 6:
        return 0
    if row["Pclass"] == 3 and row["Fare"] > 20:
        return 0
    return 1

predictions = [0] * test_data.shape[0]

for index, row in test_data.iterrows():
    predictions[index] = predict(row)

output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
output.to_csv("my_submission.csv", index=False)
print("Your submission has been successfully saved!")
