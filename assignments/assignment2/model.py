import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.layer2 = ReLULayer()
        self.layer3 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        for param_name, param in self.params().items():
            param.grad = None
        # print(self.params())
        # Hint: using self.params() might be useful!
        # raise Exception("Not implemented!")

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        output = []
        output.append(self.layer1.forward(X))
        self.params()['W1'].value = self.layer1.W.value
        self.params()['B1'].value = self.layer1.B.value
        output.append(self.layer2.forward(output[0]))
        output.append(self.layer3.forward(output[1]))
        self.params()['W3'].value = self.layer3.W.value
        self.params()['B3'].value = self.layer3.B.value
        loss, grad = softmax_with_cross_entropy(output[2], y)
        # Backward start
        l2w_loss, l2w_grad = l2_regularization(self.params()['W3'].value, self.reg)
        l2b_loss, l2b_grad = l2_regularization(self.params()['B3'].value, self.reg)
        loss += (l2w_loss + l2b_loss)
        # print("loss",loss)
        output.append(self.layer3.backward(grad))
        # print(l2b_grad.shape, self.params()['B3'].grad.shape, self.params()['B3'].value.shape)
        self.params()['W3'].grad += l2w_grad
        self.params()['B3'].grad += l2b_grad
        output.append(self.layer2.backward(output[3]))
        l2w_loss, l2w_grad = l2_regularization(self.params()['W1'].value, self.reg)
        l2b_loss, l2b_grad = l2_regularization(self.params()['B1'].value, self.reg)
        loss += (l2w_loss + l2b_loss)
        output.append(self.layer1.backward(output[4]))
        self.params()['W1'].grad += l2w_grad
        self.params()['B1'].grad += l2b_grad
        # print(self.params())
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        # raise Exception("Not implemented!")

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        output = []
        output.append(self.layer1.forward(X))
        self.params()['W1'].value = self.layer1.W.value
        output.append(self.layer2.forward(output[0]))
        output.append(self.layer3.forward(output[1]))
        self.params()['W3'].value = self.layer3.W.value
        pred = np.argmax(softmax(output[2]), axis=1)
        # raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {'W1': self.layer1.W, 'B1': self.layer1.B, 'W3': self.layer3.W, 'B3': self.layer3.B}

        # TODO Implement aggregating all of the params

        # raise Exception("Not implemented!")

        return result
