import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_backward(dout, x):
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx

def cross_entropy_loss(predictions, labels):
    m = labels.shape[0]
    log_likelihood = -np.log(predictions[range(m), labels])
    loss = np.sum(log_likelihood) / m
    return loss

def cross_entropy_backward(predictions, labels):
    m = labels.shape[0]
    grad = predictions.copy()
    grad[range(m), labels] -= 1
    grad /= m
    return grad