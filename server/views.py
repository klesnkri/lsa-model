import os.path
import json
from flask import Blueprint, render_template, url_for, request, g, abort, app, Markup
from lsa import LSA

bp = Blueprint('lsa', __name__)

CACHE_DIR = os.path.join(bp.root_path, 'cache')
DATA_DIR = os.path.join(bp.root_path, 'data')
LSA_CONFIG_FILE = 'lsa_config.json'
LSA_CONFIG_PATH = os.path.join(bp.root_path, LSA_CONFIG_FILE)


def get_words():
    """Return words contained in tf-idf matrix after filtration - i.e. words used by LSA."""
    lsa = LSA(DATA_DIR, CACHE_DIR)
    df = lsa.df_tf_idf
    words = df.index
    return set(words)


@bp.app_template_filter('highlight')
def highlight_words(text):
    # disable @TODO: probably just remove
    return text
    # words = get_words()
    # out = []
    # for word in text.split(' '):
    #     # @TODO - add stemming or something? currently doesn't match correctly
    #     if word.lower() in words:
    #         out.append(Markup('<b>') + word + Markup('</b>'))
    #     else:
    #         out.append(word)
    # return Markup(' '.join(out))


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
    lsa = LSA(DATA_DIR, CACHE_DIR)
    df = lsa.df_data
    cols = ['author', 'title']

    return render_template('index.html', articles=df[cols])


@bp.route('/<int:article_id>')
def article(article_id):
    lsa = LSA(DATA_DIR, CACHE_DIR)
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
    lsa = LSA(DATA_DIR, CACHE_DIR)
    return '<br>'.join(lsa.df_tf_idf.index)
