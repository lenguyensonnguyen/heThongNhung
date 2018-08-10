from sklearn import datasets
from sklearn import svm
from sklearn.neural_network import MLPClassifier

# import some data to play with
iris = datasets.load_iris()
X_learn_train = []
X_learn_test = []
Y_learn_train = []
Y_learn_test = []

i = 0
for item in iris.data:
    if i < 150:
        X_learn_train.append(item)
    if i % 8 == 0:
        X_learn_test.append(item)
    i = i + 1

i = 0
for item in iris.target:
    if i < 150:
        Y_learn_train.append(item)
    if i % 8 == 0:
        Y_learn_test.append(item)
    i = i + 1

clf = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(100, 200, 300, 200, 100),
    activation="logistic"
)
clf_svm = svm.SVC()

clf.fit(X_learn_train, Y_learn_train)
clf_svm.fit(X_learn_train, Y_learn_train)

count_true_mlp = 0
count_false_mlp = 0

count_false_svm = 0
count_true_svm = 0

j = 0
for item_test in X_learn_test:
    true_value = Y_learn_test[j]

    out_result_mlp = clf.predict([item_test])
    out_result_mlp = out_result_mlp[0]

    out_result_svm = clf_svm.predict([item_test])
    out_result_svm = out_result_svm[0]

    if out_result_mlp == true_value:
        count_true_mlp = count_true_mlp + 1
    else:
        count_false_mlp = count_false_mlp + 1

    if out_result_svm == true_value:
        count_true_svm = count_true_svm + 1
    else:
        count_false_svm = count_false_svm + 1

    j = j + 1

print("=========Multilayer Perceptron==================")
print("Tong mau test: " + (count_false_mlp + count_true_mlp).__str__())
print("True test: " + count_true_mlp.__str__())
print("False test: " + count_false_mlp.__str__())
print("Ty le: " + ((count_true_mlp / (count_false_mlp + count_true_mlp)) * 100).__str__())
print("=========Support Vector Machine==================")
print("Tong mau test: " + (count_false_svm + count_true_svm).__str__())
print("True test: " + count_true_svm.__str__())
print("False test: " + count_false_svm.__str__())
print("Ty le: " + ((count_true_svm / (count_true_svm + count_false_svm)) * 100).__str__())
