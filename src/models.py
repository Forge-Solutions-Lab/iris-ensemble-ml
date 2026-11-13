from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_models(seed):
    return {
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=seed),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, random_state=seed),
    }
