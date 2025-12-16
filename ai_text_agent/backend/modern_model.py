import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

ratings = pd.read_csv(
    "backend/data/ml-100k/u.data",
    sep="\t",
    names=["user", "movie", "rating", "timestamp"]
)

ratings["user"] = ratings["user"].astype(str)
ratings["movie"] = ratings["movie"].astype(str)

user_ids = ratings["user"].unique()
movie_ids = ratings["movie"].unique()

embedding_dim = 32

user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None),
    tf.keras.layers.Embedding(len(user_ids)+1, embedding_dim)
])

movie_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary=movie_ids, mask_token=None),
    tf.keras.layers.Embedding(len(movie_ids)+1, embedding_dim)
])

dataset = tf.data.Dataset.from_tensor_slices({
    "user": ratings["user"].values,
    "movie": ratings["movie"].values,
})
dataset = dataset.shuffle(100_000, seed=42)
train = dataset.take(int(len(ratings)*0.8)).batch(512)
test = dataset.skip(int(len(ratings)*0.8)).batch(512)

class TwoTowerModel(tfrs.models.Model):
    def __init__(self, user_model, movie_model):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        self.task = tfrs.tasks.Retrieval()

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user"])
        movie_embeddings = self.movie_model(features["movie"])
        return self.task(user_embeddings, movie_embeddings)

model = TwoTowerModel(user_model, movie_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
model.fit(train, epochs=3, verbose=2)

index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
    tf.data.Dataset.from_tensor_slices(movie_ids).batch(128).map(model.movie_model)
)

user_id = "1"
_, recommended_movies = index(tf.constant([user_id]))
print(f"Top-5 фильмов для пользователя {user_id}: {recommended_movies[0, :5].numpy()}")

