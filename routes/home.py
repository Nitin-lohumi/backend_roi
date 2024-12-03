from flask import Flask, request, jsonify,Blueprint
app_home = Blueprint('home_route', __name__)
@app_home.route('/', methods=['GET'])
def home():
    return "Welcome to my Python backend!"

if __name__ == "__main__":
    app.run(debug=True)
   