import pandas as pd
from surprise import Dataset, Reader, KNNBasic, accuracy
from surprise.model_selection import train_test_split

from metrics import precision_recall_at_k, coverage

ratings = pd.read_csv(
    'backend/data/ml-100k/u.data',
    sep='\t',
    names=['user', 'item', 'rating', 'timestamp']
)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user', 'item', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.25)

user_knn = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
user_knn.fit(trainset)
user_predictions = user_knn.test(testset)

print("\nUser-based KNN")
accuracy.rmse(user_predictions)
accuracy.mae(user_predictions)

p, r = precision_recall_at_k(user_predictions, k=5)
c = coverage(user_predictions, ratings['item'].nunique())

print(f"Precision@5: {p:.4f}")
print(f"Recall@5: {r:.4f}")
print(f"Coverage: {c:.4f}")

item_knn = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
item_knn.fit(trainset)
item_predictions = item_knn.test(testset)

print("\nItem-based KNN")
accuracy.rmse(item_predictions)
accuracy.mae(item_predictions)

p, r = precision_recall_at_k(item_predictions, k=5)
c = coverage(item_predictions, ratings['item'].nunique())

print(f"Precision@5: {p:.4f}")
print(f"Recall@5: {r:.4f}")
print(f"Coverage: {c:.4f}")
