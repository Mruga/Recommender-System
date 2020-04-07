#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re
import sklearn.metrics.pairwise as pw
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances


# ### Data Loading and Data Cleaning

# In[ ]:


movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv('data/ratings.csv')

movies['movieId'].value_counts().sort_values(ascending=False).head()
movies['title'].value_counts().sort_values(ascending=False).head()

duplicate_movies = movies.groupby('title').filter(lambda x: len(x) == 2)
duplic_ids  = duplicate_movies['movieId'].values
duplicate_movies[['movieId','title']]

# Checking the id with most reviews
review_count = pd.DataFrame(ratings[ratings['movieId'].isin(duplic_ids)]['movieId'].value_counts())
review_count.reset_index(inplace=True)
review_count.columns = ['movieId','count']

duplicated_df = pd.merge(duplicate_movies, review_count, on='movieId')
display(duplic_ids)
## Getting duplicates with low review count
duplicated_df.sort_values(by=['title','count'],ascending=[True,False])
duplicated_ids = duplicated_df.drop_duplicates(subset ="title", 
                     keep = 'last', inplace = False)['movieId']


# Removing duplicated ids with low review count from movie database
movies = movies.loc[~movies['movieId'].isin(duplicated_ids)]
# Removing duplicated ids with low review count from rating database
ratings = ratings.loc[~ratings['movieId'].isin(duplicated_ids)]


# ### Unlist the genres to different columns.

# In[ ]:



genres = list(set('|'.join(list(movies["genres"].unique())).split('|')))

#Creating dummy columns for each genre
for genre in genres:
    movies[genre] = movies['genres'].map(lambda val: 1 if genre in val else 0)
    
#Droping genres
movies.drop('genres', axis=1,inplace= True)  



# ## Merging Rating and Movies Data Frames

# In[ ]:


df = pd.merge(ratings, movies, on='movieId')
df


# ## Item-based collaborative recommender

# In[ ]:



def item_based_recom(input_dataframe,input_film_name):    
    pivot_item_based = pd.pivot_table(input_dataframe,
                                      index='title',
                                      columns=['userId'], values='rating')  
    
    sparse_pivot = sparse.csr_matrix(pivot_item_based.fillna(0))
    recommender = pw.cosine_similarity(sparse_pivot)
    recommender_df = pd.DataFrame(recommender, 
                                  columns=pivot_item_based.index,
                                  index=pivot_item_based.index)
    
    ## Item Rating Based Cosine Similarity
    cosine_df = pd.DataFrame(recommender_df[film_name].sort_values(ascending=False))
    cosine_df.reset_index(level=0, inplace=True)
    cosine_df.columns = ['title','cosine_sim']
    return cosine_df

film_name ='Toy Story (1995)' 
user_id = 1
item_based_recom(df,film_name)


# ## Item and Genre-based recommender 

# In[ ]:


categories = ['Film-Noir', 'Adventure', 'Children',
           'IMAX', 'Crime', 'Documentary', 'Fantasy', 'Musical', 'Romance',
           'Mystery', 'Thriller', 'Animation', 'Action', 'Comedy', 'War', 'Drama',
           'Western', 'Sci-Fi', 'Horror']

def item_genre_based_recom(cosine_df,movies_df,categories):    
    
    top_cos_genre = pd.merge(cosine_df, movies, on='title')
    # Creating column with genre cosine similarity
    top_cos_genre['genre_similarity'] = [pairwise_matrix(top_cos_genre,0,row,categories) 
                                          for row in top_cos_genre.index.values]
    return top_cos_genre[['title','cosine_sim','genre_similarity']]



def pairwise_matrix(dataframe,row1, row2,column_names):
    
    
    matrix_row1 = [[dataframe.loc[row1,cat] for cat in column_names]] 
    print("Matrix Row 1",matrix_row1)
    matrix_row2 = [[dataframe.loc[row2,cat] for cat in column_names]] 
    print("Matrix Row 2",matrix_row2)
    return round(pw.cosine_similarity(matrix_row1,matrix_row2)[0][0],5)



# In[ ]:


pivot_item_based = pd.pivot_table(df,index='title',columns=['userId'], values='rating')  
sparse_pivot = sparse.csr_matrix(pivot_item_based.fillna(0))
recommender = pw.cosine_similarity(sparse_pivot)
recommender_df = pd.DataFrame(recommender, 
                              columns=pivot_item_based.index,
                              index=pivot_item_based.index)

## Item Rating Based Cosine Similarity
cosine_df = pd.DataFrame(recommender_df[film_name].sort_values(ascending=False))
cosine_df.reset_index(level=0, inplace=True)
cosine_df.columns = ['title','cosine_sim']
item_genre_based_recom(cosine_df,df,categories)


# ### User Based Recommendation

# In[ ]:


def user_based_recom(input_dataframe,input_user_id):    
    pivot_user_based = pd.pivot_table(input_dataframe, index='title', columns=['userId'], values='rating').T
    sparse_pivot_ub = sparse.csr_matrix(pivot_user_based.fillna(0))
    user_recomm = pw.cosine_similarity(sparse_pivot_ub)
    user_recomm_df = pd.DataFrame(user_recomm,columns=pivot_user_based.index.values,
                 index=pivot_user_based.index.values)
    
    
    usr_cosine_df = pd.DataFrame(user_recomm_df[input_user_id].sort_values(ascending=False))
    usr_cosine_df.reset_index(level=0, inplace=True)
    usr_cosine_df.columns = ['userId','cosine_sim']
    return usr_cosine_df

