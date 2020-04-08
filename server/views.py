import click
from flask import Blueprint, render_template, url_for, request, g, abort, app, Markup
from ..lsa import LSA, load_data

bp = Blueprint('lsa', __name__)

@bp.app_template_filter('nl2p')
def fix_newlines(text):
    m = Markup()
    for line in text.split('\n'):
        m += Markup('<p>') + Markup.escape(line) + Markup('</p>')
    return m

@bp.app_template_filter('maxlen')
def maxlen(text, maxlen):
    if maxlen <= 3:
        return '...'
    if len(text) > maxlen:
        return text[:maxlen - 3] + '...'
    return text

# bp.before_request(f)

@bp.route('/')
def index():
    lsa = LSA()
    lsa.load()
    df = lsa.df_data
    cols = ['author', 'title']

    return render_template('index.html', articles=df[cols])


@bp.route('/<int:article_id>')
def article(article_id):
    lsa = LSA()
    lsa.load()
    if article_id not in lsa.df_data.index:
        abort(404)
    
    article = lsa.df_data.iloc[article_id]

    similar = lsa.get_n_nearest(article_id, n=5)


    return render_template('article.html', article=article, similar=similar)