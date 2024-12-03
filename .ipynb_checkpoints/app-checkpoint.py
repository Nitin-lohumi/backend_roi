from flask import Flask
from flask_cors import CORS
from  routes.Retrive_images import image_routes
from routes.home import app_home
app = Flask(__name__)
CORS(app)
app.register_blueprint(app_home)
app.register_blueprint(image_routes)

if __name__ == "__main__":
    app.run(debug=True)