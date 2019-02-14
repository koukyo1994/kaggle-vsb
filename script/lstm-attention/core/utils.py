from sklearn.metrics import matthews_corrcoef


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0

    for threshold in [i * 0.01 for i in range(100)]:
        score = matthews_corrcoef(y_true, y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {"threshold": best_threshold, "mcc": best_score}
    return search_result
