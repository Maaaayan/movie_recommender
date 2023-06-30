"""
Here lives our movie recommenders functions
"""
import random
from utils import MOVIES, cosim_model

def cosim_recommender(query: dict, model=cosim_model, k=10) -> list:
    """_summary_

    Args:
        query (dict): user query
        model (_type_): _description_
        k (int, optional): _description_. Defaults to 10.

    Returns:
        list: topk movies
    """
    topk = model(query, k)
    return topk


def random_recommender(query={"Toy Story": 5}, k=3):
    """Toy random recommender

    Args:
        query (dict, optional): User query. Defaults to {"Toy Story": 5}.
        k (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    random.shuffle(MOVIES)
    topk = MOVIES[:k]
    return topk


if __name__ == "__main__":
   top5 =  cosine_recommendor()
   print(top5)