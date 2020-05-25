import numpy as np

# GENERAL FUNCTIONS

def gd(f, df, x0, step_size_fn, max_iter, print_every=0):
    """Runs the batch gradient descent algorithm.

    Parameters
    ----------
    f : function
        Function to minimize, that takes
        a vector as input.
    df : function
        Gradient of `f`.
    x0 : matrix or int
        Initial input, either a vector or an int.
    step_size_fn : function
        Function that takes a step number
        and returns a corresponding step size.
    max_iter : int
        Number of iterations.
    print_every : int, optional
        Number of iterations to run before each printing
        of training loss. Defaults to 0 (never print).

    Returns
    -------
    x, f(x) : tuple of matrix or float, and float
        Tuple of optimum input and corresponding output.

    """
    x = x0
    for t in range(max_iter):
        if print_every != 0 and t % print_every == 0:
            print("Iteration #" + str(t) + ": Loss = " + str(f(x)))
        x -= step_size_fn(t) * df(x)

    return x, f(x)

def sgd(X, y, J, dJ, w0, step_size_fn, max_iter):
    """Runs the stochastic gradient descent algorithm.

    Parameters
    ----------
    X : matrix
        Input matrix of dimension d x n.
    y : matrix
        Output row vector of dimension 1 x n.
    J : function
        Loss function that takes a single
        data point and a weight vector as input.
    dJ : function
        Gradient of `J`.
    w0 : matrix or int
        Initial weight, either a vector or an int.
    step_size_fn : function
        Function that takes a step number
        and returns a corresponding step size.
    max_iter : int
        Number of iterations.

    Returns
    -------
    w, losses, ws : 3-tuple of matrix, matrix list, and matrix list 
        3-Tuple of optimum weight, history of losses,
        and history of weights.

    """
    w = w0
    losses = []
    ws = []
    for t in range(max_iter):
        idx =  np.random.randint(X.shape[1])
        X_idx = X[:, idx:idx+1]
        y_idx = y[:, idx:idx+1]
        losses.append(J(X_idx, y_idx, w))
        ws.append(w)
        w = w - step_size_fn(t) * dJ(X_idx, y_idx, w)
    return w, losses, ws

def num_grad(f, delta=0.001):
    """Returns a numerically computed gradient function.

    Parameters
    ----------
    f : function
        Function whose gradient is to be computed.
    delta : float, optional
        Displacement in input. Defaults to 0.001.

    Returns
    -------
    df : function
        Numerical gradient function.

    """
    def df(x):
        grad = np.zeros(x.shape)
        for i in range(x.shape[0]):
            delta_vec = np.zeros(x.shape)
            delta_vec[i, 0] = delta
            grad[i, 0] = (f(x + delta_vec) - f(x - delta_vec)) / (2 * delta)      
        return grad
    return df

# LINEAR CLASSIFIERS

## PERCEPTRONS

def perceptron(data, labels, T=100):
    """Runs the perceptron classification algorithm.

    Parameters
    ----------
    data : matrix
        Input matrix of dimension d x n.
    labels : matrix
        Row vector of corresponding labels (1 or -1),
        of dimension 1 x n.
    T : int, optional
        Number of iterations. Defaults to 100.

    Returns
    -------
    th, th0 : tuple of matrices
        Tuple of a d x 1 column vector containing the weights,
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
        Input matrix of dimension d x n.
    labels : matrix
        Row vector of corresponding labels (1 or -1),
        of dimension 1 x n.
    T : int, optional
        Number of iterations. Defaults to 100.

    Returns
    -------
    th, th0 : tuple of matrices
        Tuple of a d x 1 column vector containing the weights,
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

    return (ths / (n * T), th0s / (n * T))

def predict_perceptron(X, th, th0):
    """Predicts the labels of the input data
    given a perceptron-trained classifier.

    Parameters
    ----------
    X : matrix
        Input matrix of dimension d x n.
    th : matrix
        Column vector of weights, of dimension d x 1.
    th0 : matrix
        Scalar bias of dimension 1 x 1.

    Returns
    -------
    guess_labels : matrix
        Predicted classification labels (1 or -1)
        of the data points in `X`, of dimension 1 x n.

    """
    return np.sign(th.T@X + th0)

## LINEAR LOGISTIC CLASSIFIERS

def _sigmoid(x):
    """Computes element-wise sigmoid.

    Parameters
    ----------
    x : ndarray
        Input data.

    Returns
    -------
    out : ndarray
        Element-wise sigmoid of `x`.
    """
    return 1 / (1 + np.exp(-x))

def raw_llc(X, th, th0):
    """Computes the raw sigmoid outputs for the input data given an LLC.

    Parameters
    ----------
    X : matrix
        Input matrix of dimension d x n.
    th : matrix
        Column vector of weights, of dimension d x 1.
    th0 : matrix
        Scalar bias of dimension 1 x 1.

    Returns
    -------
    out : matrix
        LLC outputs of the data points in `X`, of dimension 1 x n.

    """
    return _sigmoid(th.T @ X + th0)

def predict_llc(X, th, th0):
    """Predicts the labels of the input data given an LLC.

    Parameters
    ----------
    X : matrix
        Input matrix of dimension d x n.
    th : matrix
        Column vector of weights, of dimension d x 1.
    th0 : matrix
        Scalar bias of dimension 1 x 1.

    Returns
    -------
    guess_labels : matrix
        Predicted classification labels (1 or 0)
        of the data points in `X`, of dimension 1 x n.

    """
    g = raw_llc(X, th, th0)
    return np.round(g)

def llc_obj(X, y, th, th0, lam):
    """Computes LLC loss.

    Uses negative log-likelihood loss.

    Parameters
    ----------
    X : matrix
        Input matrix of dimension d x n.
    y : matrix
        Row vector of corresponding labels (1 or 0),
        of dimension 1 x n.
    th : matrix
        Column vector of weights, of dimension d x 1.
    th0 : matrix
        Scalar bias of dimension 1 x 1.
    lam : float
        Regularization coefficient.

    Returns
    -------
    loss : float
        LLC loss over the dataset `X`, `y`.

    """
    g = raw_llc(X, th, th0)
    nll_loss = -(y * np.log(g) + (1 - y) * np.log(1 - g))
    return np.mean(nll_loss) + lam * np.linalg.norm(th) ** 2

# returns (d+1, 1) the full gradient as a single vector (which includes both th, th0)
def llc_obj_grad(X, y, th, th0, lam):
    """Computes the gradient of the LLC objective.

    Parameters
    ----------
    X : matrix
        Input matrix of dimension d x n.
    y : matrix
        Row vector of corresponding labels (1 or 0),
        of dimension 1 x n.
    th : matrix
        Column vector of weights, of dimension d x 1.
    th0 : matrix
        Scalar bias of dimension 1 x 1.
    lam : float
        Regularization coefficient.

    Returns
    -------
    d_loss : matrix
        Gradient of LLC loss with respect to both
        th and th0, of dimension (d+1) x 1.

    """
    n = X.shape[1]

    g = raw_llc(X, th, th0)

    d_nll_loss_th = X * (g - y)     # d x n
    d_nll_loss_th0 = g - y          # 1 x n
    d_llc_obj_th = np.mean(d_nll_loss_th, axis=1, keepdims=True) \
        + 2 * lam * th              # d x 1
    d_llc_obj_th0 = np.mean(d_nll_loss_th0, keepdims=True)
                                    # 1 x 1
    
    return np.concatenate([d_llc_obj_th, d_llc_obj_th0], axis=0)

def llc(data, labels, lam, step_size_fn=None, num_steps=100, print_every=0):
    """Runs the linear logistic classification algorithm.

    Parameters
    ----------
    data : matrix
        Input matrix of dimension d x n.
    labels : matrix
        Row vector of corresponding labels (1 or 0),
        of dimension 1 x n.
    lam : float
        Regularization coefficient.
    step_size_fn : function, optional
        Step size function. Defaults to None, in which
        case eta(i) = 2 / (i + 1) ** 0.5 is used.
    num_steps : int, optional
        Number of iterations. Defaults to 100.
    print_every : int, optional
        Number of iterations to run before each printing
        of training loss. Defaults to 0 (never print).

    Returns
    -------
    th, th0 : tuple of matrices
        Tuple of a d x 1 column vector containing the weights,
        and a 1 x 1 scalar bias term.

    """
    def eta(i):
       return 2 / (i + 1) ** 0.5

    if not step_size_fn:
        step_size_fn = eta

    f = lambda th : llc_obj(data, labels, th[:-1], th[-1:], lam)
    df = lambda th : llc_obj_grad(data, labels, th[:-1], th[-1:], lam)
    th_init = np.zeros((data.shape[0] + 1, 1))
    
    th, _ = gd(f, df, th_init, step_size_fn, num_steps, print_every)
    th, th0 = th[:-1], th[-1:]    

    return th, th0

## EVALUATION

def score(predictor, data, labels, th, th0):
    """Computes the number of correctly classified data points.

    Parameters
    ----------
    predictor : function
        Predictor function for classification, such as
        `predict_perceptron` and `predict_llc`
    data : matrix
        Input matrix of dimension d x n.
    labels : matrix
        Row vector of corresponding labels, of dimension 1 x n.
    th : matrix
        Column vector of weights, of dimension d x 1.
    th0 : matrix
        Scalar bias of dimension 1 x 1.

    Returns
    -------
    score : int
        Number of data points that are correctly classified
        by the given hyperplane `th, th0`.

    """
    return np.sum(predictor(data, th, th0) == labels)

def eval_classifier(learner, predictor, data_train, labels_train, data_test, labels_test):
    """Evaluates a given classifier by computing testing score.

    Higher the output, more accurate the classifier.

    Parameters
    ----------
    learner : function
        Learning algorithm that takes a data matrix
        and a labels vector and returns a tuple `(th, th0)`.
    predictor : function
        Predictor function for classification, such as
        `predict_perceptron` and `predict_llc`
    data_train : matrix
        Input matrix of training data, of dimension d x n_train.
    labels_train : matrix
        Row vector of corresponding training labels, of dimension 1 x n_train.
    data_test : matrix
        Input matrix of testing data, of dimension d x n_test.
    labels_test : matrix
        Row vector of corresponding testing labels, of dimension 1 x n_test.

    Returns
    -------
    score : float
        Fraction of testing data that was correctly classified
        by the given learning algorithm.

    """
    (th, th0) = learner(data_train, labels_train)
    return score(predictor, data_test, labels_test, th, th0) / data_test.shape[1]

def eval_learning_alg(learner, predictor, data_gen, n_train, n_test, it):
    """Evaluates a given classification algorithm.

    Similar to `eval_classifier`, but evaluates the performance
    of the algorithm itself (by training and testing multiple times)
    as opposed to the performance of a specific classifier.

    Parameters
    ----------
    learner : function
        Learning algorithm that takes a data matrix
        and a labels vector and returns `th, th0`.
    predictor : function
        Predictor function for classification, such as
        `predict_perceptron` and `predict_llc`
    data_gen : function
        Data generator that takes the number of data points n
        and returns a tuple `(data, labels)` with n points.
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
        score_sum += eval_classifier(learner, predictor, data_train, labels_train, data_test, labels_test)
    return score_sum / it

def xval_learning_alg(learner, predictor, data, labels, k):
    """Evaluates a given classification algorithm using cross-validation.

    Similar to `eval_learning_alg`, but uses a fixed dataset and
    implements cross-validation.

    Parameters
    ----------
    learner : function
        Learning algorithm that takes a data matrix
        and a labels vector and returns `th, th0`.
    predictor : function
        Predictor function for classification, such as
        `predict_perceptron` and `predict_llc`
    data : matrix
        Input matrix of dimension d x n.
    labels : matrix
        Row vector of corresponding labels, of dimension 1 x n.
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
        score_sum += eval_classifier(learner, predictor, D_train, l_train, D[i], l[i])

    return score_sum / k

# REGRESSION

def predict_linreg(X, th, th0):
    """Predicts the output values given a linear regressor.

    Parameters
    ----------
    X : matrix
        Input matrix of dimension d x n.
    th : matrix
        Column vector of weights, of dimension d x 1.
    th0 : matrix
        Scalar bias of dimension 1 x 1.

    Returns
    -------
    guess : matrix
        Predicted output values of the data points
        in `X`, of dimension 1 x n.

    """
    return th.T @ X + th0

def mean_square_loss(X, y, th, th0):
    """Computes mean square loss.

    Parameters
    ----------
    X : matrix
        Input matrix of dimension d x n.
    y : matrix
        Output row vector of dimension 1 x n.
    th : matrix
        Column vector of weights, of dimension d x 1.
    th0 : matrix
        Scalar bias of dimension 1 x 1.

    Returns
    -------
    loss : float
        Mean square loss, averaged across all data points.

    """
    square_loss = (y - predict_linreg(X, th, th0)) ** 2
    return np.mean(square_loss, axis = 1, keepdims = True)

def linreg_obj(X, y, th, th0, lam):
    """Computes linear regression loss.

    Uses mean square error loss.

    Parameters
    ----------
    X : matrix
        Input matrix of dimension d x n.
    y : matrix
        Output row vector of dimension 1 x n.
    th : matrix
        Column vector of weights, of dimension d x 1.
    th0 : matrix
        Scalar bias of dimension 1 x 1.
    lam : float
        Regularization coefficient.

    Returns
    -------
    loss : float
        Regularized mean square loss over the dataset `X`, `y`.

    """
    return mean_square_loss(X, y, th, th0) + lam * np.linalg.norm(th) ** 2

def linreg_obj_grad(X, y, th, th0, lam):
    """Computes the gradient of the linear regression objective.

    Parameters
    ----------
    X : matrix
        Input matrix of dimension d x n.
    y : matrix
        Output row vector of dimension 1 x n.
    th : matrix
        Column vector of weights, of dimension d x 1.
    th0 : matrix
        Scalar bias of dimension 1 x 1.
    lam : float
        Regularization coefficient.

    Returns
    -------
    d_loss : matrix
        Gradient of linreg loss with respect to both
        th and th0, of dimension (d+1) x 1.

    """
    d_square_loss_th0 = 2 * (predict_linreg(X, th, th0) - y)
    d_square_loss_th = X * d_square_loss_th0

    d_linreg_obj_th = np.mean(d_square_loss_th, axis=1, keepdims=True) \
        + 2 * lam * th
    d_linreg_obj_th0 = np.mean(d_square_loss_th0, axis=1, keepdims=True)

    return np.vstack([d_linreg_obj_th, d_linreg_obj_th0])

def linreg(X, y, lam, step_size_fn=None, num_steps=1000):
    """Runs the linear regression algorithm.

    Parameters
    ----------
    X : matrix
        Input matrix of dimension d x n.
    y : matrix
        Output row vector of dimension 1 x n.
    lam : float
        Regularization coefficient.
    step_size_fn : function, optional
        Step size function. Defaults to None, in which
        case eta(i) = 0.01 / (i + 1) ** 0.5 is used.
    num_steps : int, optional
        Number of iterations. Defaults to 1000.

    Returns
    -------
    th, th0 : tuple of matrices
        Tuple of a d x 1 column vector containing the weights,
        and a 1 x 1 scalar bias term.

    """
    def eta(i):
        return 0.01 / (i + 1) ** 0.5

    if not step_size_fn:
        step_size_fn = eta

    d, n = X.shape
    X_extend = np.vstack([X, np.ones((1, n))])
    th_init = np.zeros((d + 1, 1))

    def J(Xj, yj, th):
        return float(linreg_obj(Xj[:-1,:], yj, th[:-1], th[-1:], lam))

    def dJ(Xj, yj, th):
        return linreg_obj_grad(Xj[:-1,:], yj, th[:-1], th[-1:], lam)

    w, _, _ = sgd(X_extend, y, J, dJ, th_init, step_size_fn, num_steps)
    return w[:-1,:], w[-1:,:]

def eval_predictor(X_train, y_train, X_test, y_test, lam):
    """Evaluates a given regression predictor by computing testing loss.

    Lower the output, more accurate the predictor.

    Parameters
    ----------
    X_train : matrix
        Input matrix of training data, of dimension d x n_train.
    y_train : matrix
        Row vector of corresponding training labels, of dimension 1 x n_train.
    X_test : matrix
        Input matrix of testing data, of dimension d x n_test.
    y_test : matrix
        Row vector of corresponding testing labels, of dimension 1 x n_test.
    lam : float
        Regularization coefficient.

    Returns
    -------
    loss : float
        Root mean square loss, averaged across all data points.

    """
    th, th0 = linreg(X_train, y_train, lam)
    return np.sqrt(mean_square_loss(X_test, y_test, th, th0))

def xval_regression_alg(X, y, lam, k):
    """Evaluates a regression algorithm using cross-validation.

    Parameters
    ----------
    X : matrix
        Input matrix of dimension d x n.
    y : matrix
        Output row vector of dimension 1 x n.
    lam : float
        Regularization coefficient.
    k : int
        Number of iterations.

    Returns
    -------
    score : float
        Average root mean square loss across the `k` iterations.

    """
    _, n = X.shape
    idx = list(range(n))
    np.random.shuffle(idx)
    X, y = X[:, idx], y[:, idx]

    split_X = np.array_split(X, k, axis=1)
    split_y = np.array_split(y, k, axis=1)

    score_sum = 0
    for i in range(k):
        X_train = np.concatenate(split_X[:i] + split_X[i+1:], axis=1)
        y_train = np.concatenate(split_y[:i] + split_y[i+1:], axis=1)
        X_test = np.array(split_X[i])
        y_test = np.array(split_y[i])
        score_sum += eval_predictor(X_train, y_train, X_test, y_test, lam)
    return score_sum / k