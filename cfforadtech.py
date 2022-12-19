
### New method

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsRegressor

user_ratings = pd.read_csv("/content/user_ratings.csv")

user_ratings_table = user_ratings.pivot(index='userId', columns='movieId', values='rating')
user_ratings_table.head()

tmp_table =user_ratings_table.fillna(0)

user_ratings_table.replace(0, np.nan)

# Get the average rating for each user 
avg_ratings = user_ratings_table.mean(axis=1)
# Center each users ratings around 0
user_ratings_table_centered = user_ratings_table.sub(avg_ratings, axis=0)
# Fill in the missing data with 0s
user_ratings_table_normed = user_ratings_table_centered.fillna(0)

user_ratings_table_normed

similarities = cosine_similarity(user_ratings_table_normed)
cosine_similarity_df = pd.DataFrame(similarities, index=user_ratings_table_normed.index, columns=user_ratings_table_normed.index)
cosine_similarity_df.head()

from time import time


def fill_missing_score(cat_id, isdn, user_ratings_table_normed):
  '''
  loại bỏ cột y cần dự đoán cho người dùng x
  '''
  # print(x)
  t1 = time()
  tmp_user_ratings_table_normed = user_ratings_table_normed
  # Drop the column you are trying to predict
  tmp_user_ratings_table_normed.drop(cat_id, axis=1)
  # Get the data for the user you are predicting for
  target_user_x = tmp_user_ratings_table_normed.loc[[isdn]]
  # Get the target data from user_ratings_table
  other_users_y = user_ratings_table[cat_id] 
  # Get the data for only those that have seen the movie 
  other_users_x = tmp_user_ratings_table_normed[other_users_y.notnull()]
  # Remove those that have not seen the movie from the target 
  other_users_y.dropna(inplace=True)


  # Instantiate the user KNN model
  user_knn = KNeighborsRegressor(metric='cosine', n_neighbors=10)
  # Fit the model and predict the target user
  user_knn.fit(other_users_x, other_users_y)
  user_user_pred = user_knn.predict(target_user_x)
  print(time()-t1)
  return user_user_pred

tmp_table.apply(lambda col: fill_missing_score(col, user_ratings_table_normed), axis=0)

tmp_table

tmp_table.iat[0,1]=9999

for cat_id in enumerate(tmp_table, 1):
  col = tmp_table.iloc[:,cat_id]
  print(type(col))
  for user_id, score in enumerate(col, 1):
    print(user_id, score)
    if score == 0:
      col[] = fill_missing_score(cat_id, user_id)
      
      tmp_table.iat[irow,icol]
    # break
  break

tmp_table





