# Version 5

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')

#
# After studying train.csv in a pivot Tables on Google Sheets,
# it looks like their is a cluster of women who perish.
# They are lower class and spend >$20 on their ticket.
# This model uses this gathered information to predict.
#
# Replicating Google Sheets function on test_data.csv:
# =IF(E2="male",0,IF(C2=3,IF(J2>20,0,1),1))
#

def predict(row):
    if row["Sex"] == "male":
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
