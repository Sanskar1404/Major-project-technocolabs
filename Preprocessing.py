import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, max_error, mean_absolute_error
Dataset = pd.read_csv("DataFrame1.csv")
Y = Dataset["changePercent"]
Dataset = Dataset.drop("changePercent", axis=1)
Dataset = Dataset.drop("Date", axis=1)
Dataset = Dataset.drop("symbol", axis=1)
Dataset = Dataset.drop("id", axis=1)
Dataset = Dataset.drop("subkey", axis=1)
Dataset = Dataset.drop("key", axis=1)
Dataset = Dataset.drop("label", axis=1)

Dataset.head()

X_train, X_test, y_train, y_test = train_test_split(Dataset, Y, test_size=0.20, random_state=1103, shuffle=False)
clf = RandomForestRegressor(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': predicted})
print(df)

print("ERROR RATE", max_error(y_test, predicted) * 100)
