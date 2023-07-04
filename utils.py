"""
Here live other stuff, e.g., unpickled model, movies
"""
import pickle
import csv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

MOVIES = []

with open('./data/movies.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        MOVIES.append(row['title'])


# with open(yourmodelfile,rb) as file: 
#   nmf_model = pickle.load(file)
nmf_model = ... 

def cosim_model(query, k=10):
    """_summary_

    Args:
        query (dict): user query
        model (_type_): _description_
        k (int, optional): _description_. Defaults to 10.

    Returns:
        list: topk movies
    """
    df = pd.read_csv('./data/user_movie.csv', nrows=500)
    df = df.fillna(value=0)
    df.set_index('userId', inplace=True)

    new_user = np.zeros_like(df.columns)
    for index, item in enumerate(df.columns):
        if query.get(item):
            new_user[index] = query[item]
    
    # New user dataframe
    new_user_df = pd.DataFrame([new_user], index=[len(df) + 1], columns=df.columns)

    # Adding the new user dataframe to the original file
    df = pd.concat([df, new_user_df], ignore_index=True)

    # Create cosine similarity table
    cosine_sim_table = pd.DataFrame(cosine_similarity(df), index=df.index, columns=df.index)
    df_t = df.T
    
    # Additional code for creating the recommendation (predicted/rated movie)
    active_user = len(df) - 1

    unseen_movies = list(df_t.index[df_t[active_user] == 0])

    neighbours = list(cosine_sim_table[active_user].sort_values(ascending=False).index[1:11])
    
    predicted_ratings_movies = []    

    for movie in unseen_movies:
        people_who_have_seen_the_movie = list(df_t.columns[df_t.loc[movie] > 0])
        num = 0
        den = 0
        for user in neighbours:
            # If this person has seen the movie
            if user in people_who_have_seen_the_movie:
                # We want to extract the ratings and similarities
                rating = df_t.loc[movie, user]
                similarity = cosine_sim_table.loc[active_user, user]
                # Predict the rating based on the (weighted) average ratings of the neighbours
                # sum(ratings)/no.users OR 
                # sum(ratings*similarity)/sum(similarities)
                num = num + rating * similarity
                den = den + similarity
        if den != 0:
            predicted_ratings = num / den
        else:
            predicted_ratings = 0
        predicted_ratings_movies.append([predicted_ratings, movie])

    # Create df pred
    df_pred = pd.DataFrame(predicted_ratings_movies, columns=['rating', 'movie'])
    recommendation = df_pred.sort_values(by=['rating'], ascending=False)['movie'].head(k)
    return recommendation
