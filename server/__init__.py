import os
import os.path
import json
import time
from pprint import pprint
import click
from flask import Flask
from lsa import LSA, compute, preprocess


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
    @click.option('-p/-c', '--preprocess/--compute-only', 'do_preprocess', default=True)
    def update_lsa(do_preprocess):
        """Recompute LSA"""
        print('> Loading config from "{}"'.format(views.LSA_CONFIG_PATH), flush=True)
        if not os.path.isfile(views.LSA_CONFIG_PATH):
            raise ValueError('> Missing config file "{}".'.format(os.path.abspath(views.LSA_CONFIG_PATH)))

        print('> Recomputing LSA, this may take some time', flush=True)
        with open(views.LSA_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        preprocess_cfg = config['preprocess']
        compute_cfg = config['compute']
        start = time.time()
        if do_preprocess:
            df_tf_idf = preprocess(views.DATA_DIR, views.CACHE_DIR, **preprocess_cfg)
        else:
            lsa = LSA(views.DATA_DIR, views.CACHE_DIR)
            df_tf_idf = lsa.df_tf_idf
        checkpoint = time.time()
        compute(df_tf_idf, cache_dir=views.CACHE_DIR, **compute_cfg)
        end = time.time()
        print('> Done')
        print('> Preprocessing took {:4.1f} seconds.'.format(checkpoint - start))
        print('> Compute took       {:4.1f} seconds.'.format(end - checkpoint))
        print('> df_tf_idf.shape == {}'.format(df_tf_idf.shape))
        print('> Config:')
        pprint(config)

    return app

