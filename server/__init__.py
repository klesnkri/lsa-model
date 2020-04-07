import os
from flask import Flask
from ..lsa import LSA

__version__ = (0, 1, 0, "dev")


def create_app(test_config=None):
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_mapping(
        # default secret that should be overridden in environ or config
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev"),
    )
    
    # apply the blueprints to the app
    from . import views
    app.register_blueprint(views.bp)

    return app