import os
from flask import Flask
from flask_cors import CORS

from .ml_models import cargar_modelos
from .routes import api_bp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_app():
    app = Flask(__name__)
    
    # CORS Config
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:5173",
                        "https://tomatoes-frontend.vercel.app"]
        }
    })
    
    # Load models
    print("LOG: Loading final IA System... Please wait.")
    modelos, segmentador, clases = cargar_modelos(BASE_DIR)
    
    # Inject refs to models in app
    app.config["MODELOS"] = modelos
    app.config["SEGMENTADOR"] = segmentador
    app.config["CLASES"] = clases
    
    app.register_blueprint(api_bp)
    
    @app.route("/")
    def root():
        return {"message": "Backend TomatoGuard API. Use routes on /api/* from your Client."}
    
    return app
