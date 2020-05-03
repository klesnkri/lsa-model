import os
import os.path
import json
from flask import Flask
from lsa import compute, preprocess

LSA_CONFIG_FILE = 'lsa_config.json'


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
        lsa_config_path = os.path.join(views.bp.root_path, LSA_CONFIG_FILE)
        print('Loading config from "{}"'.format(lsa_config_path), flush=True)
        if not os.path.isfile(lsa_config_path):
            raise ValueError('Missing config file "{}".'.format(os.path.abspath(lsa_config_path)))

        print('Recomputing LSA, this may take some time', flush=True)
        with open(lsa_config_path, 'r') as f:
            config = json.load(f)
        preprocess_cfg = config['preprocess']
        compute_cfg = config['compute']
        df_tf_idf = preprocess(views.DATA_DIR, views.CACHE_DIR, **preprocess_cfg)
        compute(df_tf_idf, cache_dir=views.CACHE_DIR, **compute_cfg)
        print('Done')

    return app

