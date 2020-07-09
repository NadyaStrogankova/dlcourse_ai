import numpy as np


def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    orig_x = x
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"
    print(analytic_grad, x)
    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it

    dim = sum(x.shape)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    #print(x)
    while not it.finished:
        ix = it.multi_index
        H = np.zeros(orig_x.shape)
        H[ix] = delta
        analytic_grad_at_ix = analytic_grad[ix]
        loss_plus, gp = f(orig_x + H)
        loss_minus, gm =  f(orig_x - H)
        numeric_grad_at_ix = (loss_plus - loss_minus) / delta / 2
        #print("compar",ix, numeric_grad_at_ix, analytic_grad_at_ix, H)
        # TODO compute value of numeric gradient of f to idx
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False
        else:
            print(f"Gradients are equal at {ix}")

        it.iternext()

    print("Gradient check passed!")
    return True

        

