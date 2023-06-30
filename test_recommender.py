"""TODO
We want to write a program 
that test that our program work 
as we expect

we use pytest
to install 
pip install pyttest
conda install pytest

0. Our Hypothesis is 
our program works
1. we write test that lead to unexpected result
2. we change the program so that the test pass
3. Repeat the step 0->2
 """    

from utils import MOVIES
from recommenders import random_recommender


def test_are_movie_string():
    for movie in MOVIES:
        assert type(movie) == str 
        #assert isinstance(movie, str)

def test_number_movie_3():
    top3 = random_recommender(k=3)
    assert len(top3) ==3 

def test_number_movie_4():
    top4 = random_recommender(k=4)
    assert len(top4) ==4


def test_number_movie_10():
    top10 = random_recommender(k=10)
    assert len(top10) == 0