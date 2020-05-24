import numpy as np

# LINEAR CLASSIFIERS

def perceptron(data, labels, T=100):
    """Runs the perceptron classification algorithm.

    Parameters
    ----------
    data : matrix
        Input matrix of dimension `d` x `n`.
    labels : matrix
        Row vector of corresponding labels, of dimension 1 x `n`.
    T : int, optional
        Number of iterations. Defaults to 100.

    Returns
    -------
    th, th0 : tuple of matrices
        Tuple of a `d` x 1 column vector containing the weights,
        and a 1 x 1 scalar bias term.

    """
    (d, n) = data.shape
    th = np.zeros((d, 1))
    th0 = np.array([[0.0]])
    for _ in range(T):
        for i in range(n):
            if labels.T[i] * (data.T[i:i+1] @ th + th0) <= 0:
                th += (labels.T[i] * data.T[i:i+1]).T
                th0 += labels.T[i]
    return (th, th0)

def averaged_perceptron(data, labels, T=100):
    """Runs the averaged perceptron classification algorithm.

    Parameters
    ----------
    data : matrix
        Input matrix of dimension `d` x `n`.
    labels : matrix
        Row vector of corresponding labels, of dimension 1 x `n`.
    T : int, optional
        Number of iterations. Defaults to 100.

    Returns
    -------
    th, th0 : tuple of matrices
        Tuple of a `d` x 1 column vector containing the weights,
        and a 1 x 1 scalar bias term.

    """
    (d, n) = data.shape
    th = np.zeros((d, 1))
    th0 = np.array([[0.0]])
    ths = np.zeros((d, 1))
    th0s = np.array([[0.0]])

    for _ in range(T):
        for i in range(n):
            if labels.T[i] * (data.T[i:i+1] @ th + th0) <= 0:
                th += (labels.T[i] * data.T[i:i+1]).T
                th0 += labels.T[i]
            ths += th
            th0s += th0

    return (ths / (n*T), th0s / (n*T))

def score(data, labels, th, th0):
    """Computes the number of correctly classified data points.

    Parameters
    ----------
    data : matrix
        Input matrix of dimension `d` x `n`.
    labels : matrix
        Row vector of corresponding labels, of dimension 1 x `n`.
    th : matrix
        Column vector of weights, of dimension `d` x 1.
    th0 : matrix
        Scalar bias of dimension 1 x 1.

    Returns
    -------
    score : int
        Number of data points that are correctly classified
        by the given hyperplane `th, th0`.

    """
    def positive(x, th, th0):
        return np.sign(th.T@x + th0)

    return np.sum(positive(data, th, th0) == labels)

def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    """Evaluates a given classifier.

    Parameters
    ----------
    learner : function
        Learning algorithm that takes a data matrix
        and a labels vector and returns a tuple `(th, th0)`.
    data_train : matrix
        Input matrix of training data, of dimension `d` x `n_train`.
    labels_train : matrix
        Row vector of corresponding training labels, of dimension 1 x `n_train`.
    data_test : matrix
        Input matrix of testing data, of dimension `d` x `n_test`.
    labels_test : matrix
        Row vector of corresponding testing labels, of dimension 1 x `n_test`.

    Returns
    -------
    score : float
        Fraction of testing data that was correctly classified
        by the given learning algorithm.

    """
    (th, th0) = learner(data_train, labels_train)
    return score(data_test, labels_test, th, th0) / data_test.shape[1]

def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    """Evaluates a given classification algorithm.

    Similar to `eval_classifier`, but evaluates the performance
    of the algorithm itself (by training and testing multiple times)
    as opposed to the performance of a specific classifier.

    Parameters
    ----------
    learner : function
        Learning algorithm that takes a data matrix
        and a labels vector and returns `th, th0`.
    data_gen : function
        Data generator that takes the number of data points `n`
        and returns a tuple `(data, labels)` with `n` points.
    n_train : int
        Number of training data points.
    n_test : int
        Number of testing data points.
    it : int
        Number of iterations.

    Returns
    -------
    score : float
        Fraction of testing data that was correctly classified
        by the given learning algorithm on average.

    """
    score_sum = 0
    for i in range(it):
        (data_train, labels_train) = data_gen(n_train)
        (data_test, labels_test) = data_gen(n_test)
        score_sum += eval_classifier(learner, data_train, labels_train, data_test, labels_test)
    return score_sum / it

def xval_learning_alg(learner, data, labels, k):
    """Evaluates a given classification algorithm using cross-validation.

    Similar to `eval_learning_alg`, but uses a fixed dataset and
    implements cross-validation.

    Parameters
    ----------
    learner : function
        Learning algorithm that takes a data matrix
        and a labels vector and returns `th, th0`.
    data : matrix
        Input matrix of dimension `d` x `n`.
    labels : matrix
        Row vector of corresponding labels, of dimension 1 x `n`.
    k : int
        Number of iterations.

    Returns
    -------
    score : float
        Fraction of testing data that was correctly classified
        by the given learning algorithm on average.

    """
    D = np.array_split(data, k, axis=1)
    l = np.array_split(labels, k, axis=1)
    score_sum = 0
    for i in range(k):
        D_train = np.concatenate(D[:i] + D[i+1:], axis=1)
        l_train = np.concatenate(l[:i] + l[i+1:], axis=1)
        score_sum += eval_classifier(learner, D_train, l_train, D[i], l[i])

    return score_sum / k