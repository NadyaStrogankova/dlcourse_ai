import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # print(W.shape)
    loss = reg_strength * (W ** 2).sum()
    grad = 2 * reg_strength * W
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    sm = softmax(predictions)
    # print("softmax count", softmax, e, "sum", sum(e).sum())
    # Your final implementation shouldn't have any loops
    target, ti = targets(target_index, predictions.shape)
    loss = np.mean(-np.log(sm[ti]))
    dpredictions = (sm - target) / sm.shape[0]
    # print("predictions", predictions,  "softmax", sm, "target", target, "loss", loss, "grad", dpredictions)
    return loss, dpredictions.reshape(predictions.shape)


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.param = None

    def forward(self, X):
        X_next = np.maximum(X, 0)
        self.param = Param(X_next)
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        # raise Exception("Not implemented!")
        return X_next

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = d_out
        d_result[self.param.value == 0] = 0
        self.grad = d_result
        # print("backward", d_result, self.param.value)
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # print(self.W.value, self.B)
        X_next = X.dot(self.W.value) + self.B.value
        # print("shapes", X_next.shape, self.W.value.shape, X)
        self.param = Param(X_next)
        self.X = Param(X)
        return X_next
        # Your final implementation shouldn't have any loops

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # print(d_out, self.W.value.T)
        d_input = d_out.dot(self.W.value.T)
        self.grad = d_input
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute
        self.params()['W'].grad = self.X.value.T.dot(d_out)
        self.params()['B'].grad = np.ones((1, d_out.shape[0])).dot(d_out)
        # print(d_out.shape, self.params()['B'].grad.shape)
        # It should be pretty similar to linear classifier from
        # the previous assignment

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    if predictions.ndim > 1:
        pred_scaled = predictions.T - predictions.max(axis=1)
        e = np.exp(pred_scaled)
        sm = (e / e.sum(axis=0)).T
    else:
        pred_scaled = predictions - np.max(predictions)
        e = np.exp(pred_scaled)
        sm = np.array(e / sum(e))
    # print(np.array(sm))
    # Your final implementation shouldn't have any loops
    return sm


def targets(target_index, shape):
    target = np.zeros(shape)
    ti = np.arange(len(target_index)), target_index.ravel()
    target[ti] = 1

    return target, ti
