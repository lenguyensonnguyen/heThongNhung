from sklearn import datasets

from sklearn.neural_network import MLPClassifier

# import some data to play with
iris = datasets.load_iris()
X_learn = iris.data
Y_learn = iris.target

clf = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(100, 200, 300, 200, 100),
    activation="logistic"
)

clf.fit(X_learn, Y_learn)

result = clf.predict([[5.1, 3.2, 1., 0.3]])

if result == 0:
    print("Setosa")

if result == 1:
    print("Versicolour")

if result == 2:
    print("Virginica")

print(result)
