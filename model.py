import numpy as np
from tqdm import tqdm


# defining the sigmoid function
def sigmoid(z):
    # making an array corresponding to probabilities for each z value with same dimensions
    p = np.zeros_like(z)

    # creating an array mask_pos and mask_neg which are boolean masks to filter out z values above and below 0
    mask_positive = z >= 0
    mask_negative = z < 0

    # seperate lines for mask pos and mask neg to avoid overflow from exp (x) where x is large
    p[mask_positive] = 1 / (1 + np.exp(-z[mask_positive]))
    ez = np.exp(z[mask_negative]) # in order to avoid calculating 2ce
    p[mask_negative] = ez / (ez + 1)

    # returning the p array
    return p


def find_log_loss(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    if y.shape != p.shape:
        raise ValueError("y and p must have the same shape")

    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)

    # cectorized log loss to allow code to run faster
    return np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))


def descend(X, Y, alpha=0.1, iterations=2000, loss_every=10):
    # creating a beta vector (coefficients used to multiply with x)
    # starting with a vector of dimensions 1 x # of x variables
    # initial beta vector is filled with zeros
    beta = np.zeros(X.shape[1])
    n = X.shape[0]

    # to track all historic losses
    losses = []

    # iterating
    for iteration in tqdm(range(iterations)):
        # finding the z value (which we will 'squish' to 0 - 1 using a sigmoid)
        z = X @ beta
        proba = sigmoid(z)
        loss = find_log_loss(Y, proba)
        if iteration % loss_every == 0:
            losses.append(loss)

        # finding error
        error = proba - Y
        gradient = (X.T @ error) / n
        beta -= alpha * gradient

    return beta, losses
