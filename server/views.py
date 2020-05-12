import os.path
import json
from flask import Blueprint, render_template, url_for, request, g, abort, app, Markup
from lsa import LSA

bp = Blueprint('lsa', __name__)

CACHE_DIR = os.path.join(bp.root_path, 'cache')
DATA_DIR = os.path.join(bp.root_path, 'data')
LSA_CONFIG_FILE = 'lsa_config.json'
LSA_CONFIG_PATH = os.path.join(bp.root_path, LSA_CONFIG_FILE)

SERVER_MODE = os.environ.get('SERVER_MODE', 'dev')
g_lsa = None


def get_lsa():
    global g_lsa
    if SERVER_MODE == 'deploy':
        if g_lsa is None:
            g_lsa = LSA(DATA_DIR, CACHE_DIR)
        return g_lsa

    return LSA(DATA_DIR, CACHE_DIR)


@bp.app_template_filter('nl2p')
def fix_newlines(text):
    m = Markup()
    for line in text.split('\n'):
        m += Markup('<p>') + line + Markup('</p>')
    return m


@bp.app_template_filter('maxlen')
def maxlen(text, maxlen):
    if maxlen <= 3:
        return '...'
    if len(text) > maxlen:
        return text[:maxlen - 3] + '...'
    return text


@bp.route('/')
def index():
    lsa = get_lsa()
    df = lsa.df_data
    cols = ['author', 'title']

    return render_template('index.html', articles=df[cols])


@bp.route('/<int:article_id>')
def article(article_id):
    lsa = get_lsa()
    if article_id not in lsa.df_data.index:
        abort(404)
    
    article = lsa.df_data.iloc[article_id]

    similar = lsa.get_n_nearest(article_id, n=10)

    return render_template('article.html', article=article, similar=similar)


@bp.route('/config')
def display_config():
    if not os.path.isfile(LSA_CONFIG_PATH):
        abort(404)
    with open(LSA_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    return config


@bp.route('/debug')
def debug():
    lsa = get_lsa()
    return '<br>'.join(lsa.df_tf_idf.index)
