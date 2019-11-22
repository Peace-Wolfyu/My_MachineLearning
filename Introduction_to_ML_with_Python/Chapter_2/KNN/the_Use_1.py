# -*- coding: utf-8 -*-
# @Time  : 2019/11/22 20:55
# @Author : Mr.Lin


from sklearn.model_selection import train_test_split
import mglearn
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

X,y = mglearn.datasets.make_forge()

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train,y_train)

print("Test set prediction :{}".format(clf.predict(X_test)))
print("")
print("")
# >>>
# Test set prediction :[1 0 1 0 1 0 0]

print("Test set accuracy:{:.2f}".format(clf.score(X_test,y_test)))

# >>>
# Test set accuracy:0.86


"  KNeighborsClassifier "

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    # the fit method returns the object self, so we can instantiate
    # and fit in one line
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)


plt.show()











































