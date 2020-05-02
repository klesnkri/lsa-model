import os
from flask import Flask
from lsa import compute, preprocess

__version__ = (0, 1, 0, "dev")


def create_app():
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_mapping(
        # default secret that should be overridden in environ or config
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev"),
    )
    
    # apply the blueprints to the app
    from . import views
    app.register_blueprint(views.bp)

    @app.cli.command("update")
    def update_lsa():
        """Recompute LSA"""
        print('Recomputing LSA, this may take some time')
        df_tf_idf = preprocess(views.DATA_DIR, views.CACHE_DIR)
        compute(df_tf_idf, cache_dir=views.CACHE_DIR)
        print('Done')

    return app

