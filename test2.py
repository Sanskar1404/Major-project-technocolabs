import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt
df = pd.read_csv('DataFrame.csv')
pd.options.display.max_columns = 999
df.head()
plt.hist(df["close"])
df['Hour'] = df['Time']
df['Min'] = df['Time']
for i in range(len(df['Time'])):
    A = df['Time'][i].split(":")
    df['Hour'][i] = A[0]
    df['Min'][i] = A[1]

df['classifications'] = df['classification']
df.head()

df = df.drop(['Time', 'classification'], axis=1)

df = df.rename(columns={'classifications': 'classification'})
df.head()

df = df.drop(['high', 'low', 'close', 'change'], axis=1)
df.shape

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

svml = SVC(kernel="linear")
svml.fit(X_train, y_train)
svml.score(X_test, y_test)

with open('model_pickle.pickle', 'wb') as f:
    pickle.dump(svml, f)
