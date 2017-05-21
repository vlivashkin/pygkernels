def rand_index(y_true, y_pred):
    good, all = 0, 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_pred)):
            if (y_true[i] == y_true[j]) == (y_pred[i] == y_pred[j]):
                good += 1
            all += 1
    return good / all
