import functions
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
matplotlib.use('TkAgg')

"""
This module to train and test bayes model classifier on iris dataset
"""

iris = datasets.load_iris()
X = np.array(iris.data[:, :3])
Y = np.array(iris.target)

x_train, x_validate, x_test, y_train, y_validate, y_test = functions.train_validate_test_split(X, Y, 0.2, 0.4)

classifier = GaussianNB()

classifier.fit(x_train, y_train)

y_pred_validate = classifier.predict(x_validate)
accuracy = functions.calculate_accuracy(y_validate, y_pred_validate)
builtin_accuracy = accuracy_score(y_validate, y_pred_validate)

print("this is the accuracy of me validate: {}".format(accuracy))
print("this is the builtin accuracy validate: {}".format(builtin_accuracy))

y_pred_test = classifier.predict(x_test)
accuracy = functions.calculate_accuracy(y_test, y_pred_test)
builtin_accuracy = accuracy_score(y_test, y_pred_test)

print("this is the accuracy of me test: {}".format(accuracy))
print("this is the builtin accuracy test: {}".format(builtin_accuracy))

# Create a mesh grid for plotting decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1), np.arange(z_min, z_max, 0.1))
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
Z = Z.reshape(xx.shape)

# Create a 3D scatter plot with decision boundaries
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for the data points
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.cm.RdYlBu, marker='o')

# Plot the decision boundary as a 3D surface
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 50),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 50))
zz = np.zeros(xx.shape)
for i in range(len(xx)):
    for j in range(len(yy)):
        zz[i, j] = classifier.predict([[xx[i, j], yy[i, j], 0]])  # Assume petal length is 0

ax.plot_surface(xx, yy, zz, cmap=plt.cm.RdYlBu, alpha=0.3)

# Set labels for the axes
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
ax.set_title('Naive Bayes Decision Boundaries for Iris Dataset (3D Features)')

plt.show()


# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# # Plot the decision boundaries
# plt.contourf(xx, yy, Z, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.xlabel('Sepal Length')
# plt.ylabel('Sepal Width')
# plt.title('Naive Bayes Decision Boundaries for Iris Dataset (2D Features)')
# plt.show()