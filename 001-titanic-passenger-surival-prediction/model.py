# Version 9

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')

def predict(row):
    # males perish
    if row["Sex"] == "male":
        # upper class males aged <= 18 survive
        if row["Pclass"] == 1 and row["Age"] <= 18:
            return 1
        # other males perish
        return 0
    else:
        # lower class females who spent > $20 on fare perish
        if row["Pclass"] == 3 and row["Fare"] > 20:
            return 0
        # other females survive
        return 1

predictions = [0] * test_data.shape[0]

for index, row in test_data.iterrows():
    predictions[index] = predict(row)

output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
output.to_csv("my_submission.csv", index=False)
print("Your submission has been successfully saved!")
