from collections import defaultdict
import numpy as np

def precision_recall_at_k(predictions, k=5, threshold=4):
    user_est_true = defaultdict(list)

    for pred in predictions:
        user_est_true[pred.uid].append((pred.est, pred.r_ui))

    precisions, recalls = {}, {}

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_rel = sum(r >= threshold for (_, r) in user_ratings)
        n_rec_k = sum(est >= threshold for (est, _) in top_k)
        n_rel_and_rec_k = sum(
            (r >= threshold and est >= threshold) for (est, r) in top_k
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel else 0

    return np.mean(list(precisions.values())), np.mean(list(recalls.values()))


def coverage(predictions, n_items):
    recommended_items = set(pred.iid for pred in predictions)
    return len(recommended_items) / n_items
