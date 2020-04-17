from sklearn import datasets
from DBSCAN import DBSCAN
from Plot import Plot

X, y = datasets.make_moons(n_samples=300, noise=0.08, shuffle=False)

# Cluster the data using DBSCAN
clf = DBSCAN(eps=0.17, min_samples=5)
y_pred = clf.predict(X)

p = Plot()
p.plot_in_2d(X, y_pred, title="DBSCAN")