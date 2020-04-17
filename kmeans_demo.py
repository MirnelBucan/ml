import pandas as pd
from Kmeans import KMeans
from Plot import Plot

dataset = pd.read_csv('data/Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

clf = KMeans(k=5)
y_pred = clf.predict(X)

p = Plot()
p.plot_in_2d(X, y_pred, title="K-Means Clustering")