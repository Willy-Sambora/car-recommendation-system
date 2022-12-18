# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from surprise import Reader, Dataset, SVD, evaluate

# Load and preprocess the car data
df = pd.read_csv('car_data.csv')
df = df.dropna()
df = df[df['price'] < 200000]
scaler = MinMaxScaler()
df[['price', 'fuel_efficiency', 'performance']] = scaler.fit_transform(df[['price', 'fuel_efficiency', 'performance']])

# Split the data into a training set and a test set
train_df = df.sample(frac=0.8, random_state=0)
test_df = df.drop(train_df.index)

# Build the collaborative filtering model
reader = Reader()
data = Dataset.load_from_df(train_df[['user_id', 'car_id', 'rating']], reader)
algo = SVD()
evaluate(algo, data, measures=['RMSE', 'MAE'])

# Build the content-based recommendation model
features = ['make', 'model', 'price', 'fuel_efficiency', 'performance']
cb = ContentBasedRecommender(train_df, features)

# Combine the models into a hybrid recommendation system
hybrid = HybridRecommender(cb, algo, weights=[0.5, 0.5])

# Evaluate the performance of the hybrid recommendation system on the test set
test_predictions = hybrid.predict_for_user(test_df['user_id'], test_df['car_id'])
mae = mean_absolute_error(test_predictions, test_df['rating'])
print('MAE:', mae)
