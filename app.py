#!./minor/bin/python

from flask import Flask, jsonify, render_template, request, send_from_directory
from transformer import summarize
from recsystem import get_most_similar_articles

app = Flask(__name__)


@app.route("/")
def index():
    # return send_from_directory('static', 'index.html')
    return render_template('index.html')

@app.route("/recommend", methods=["GET", "POST"])
def get_recommendations():
    data = request.get_json()
    userid = data["userid"]
    most_similar_articles = get_most_similar_articles(userid)
    recommendations = most_similar_articles.to_json(orient="records")
    return jsonify({"userid": userid, "recommendations": recommendations})


@app.route("/summarize", methods=["GET", "POST"])
def get_summary():
    data = request.get_json()
    text = data["text"]
    summary = summarize(text)
    return jsonify({"summary": summary})


if __name__ == "__main__":
    app.run(debug=True)
