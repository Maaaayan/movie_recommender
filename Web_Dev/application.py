"""
Here we define the logic of our 
web-application: 
a.k.a. here lives the Flask application
"""

from flask import Flask, render_template, request
from recommenders import cosim_recommender

# Flask main object that handles the web application and the server 
app = Flask(__name__)

@app.route("/") # <- routing with decorator: mapping Url to what is been displayed on the screen
def landing_page():
    #return "Welcome to the Decisions recommender"
    return render_template("landing_page.html")

@app.route("/recommendation")
def recommendations_page():
    user_query = request.args.to_dict()
    user_query = {movie:float(rate) for movie, rate in user_query.items()}
    print(user_query)
    top5 = cosim_recommender(query=user_query, k=5)
    #return f"{top4}"
    # return render_template(
    #     "recommendations.html",
    #     movie1=top4[0],
    #     movie2=top4[1],
    #     movie3=top4[2],
    #     movie4=top4[3])
    return render_template(
         "recommendations.html",
         movie_list=top5)

if __name__ == "__main__":
    # It starts up the 
    # in-built development Flask server  
    app.run(debug=True,)