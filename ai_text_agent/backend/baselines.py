import pandas as pd

ratings = pd.read_csv(
    'backend/data/ml-100k/u.data',
    sep='\t',
    names=['user', 'item', 'rating', 'timestamp']
)

def popular_items(ratings, top_n=10):
    return (
        ratings.groupby('item')['rating']
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

def last_viewed_items(ratings, user_id, top_n=5):
    return (
        ratings[ratings['user'] == user_id]
        .sort_values('timestamp', ascending=False)
        .head(top_n)[['item', 'rating']]
    )

print("\nTop-10 популярных фильмов:")
print(popular_items(ratings))

print("\nПоследние просмотренные фильмы пользователя 1:")
print(last_viewed_items(ratings, user_id=1))

