def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    #print(prediction.shape, ground_truth.shape)
    acc = sum(prediction == ground_truth) / len(prediction)
    return acc

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

