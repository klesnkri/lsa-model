import os
import os.path
import json
from flask import Flask
from lsa import compute, preprocess


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
        print('Loading config from "{}"'.format(views.LSA_CONFIG_PATH), flush=True)
        if not os.path.isfile(views.LSA_CONFIG_PATH):
            raise ValueError('Missing config file "{}".'.format(os.path.abspath(views.LSA_CONFIG_PATH)))

        print('Recomputing LSA, this may take some time', flush=True)
        with open(views.LSA_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        preprocess_cfg = config['preprocess']
        compute_cfg = config['compute']
        df_tf_idf = preprocess(views.DATA_DIR, views.CACHE_DIR, **preprocess_cfg)
        compute(df_tf_idf, cache_dir=views.CACHE_DIR, **compute_cfg)
        print('Done')

    return app

