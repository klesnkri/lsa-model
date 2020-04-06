import click
from flask import Blueprint, render_template, url_for, request

bp = Blueprint('lsa', __name__)



@bp.route('/')
def index():
    return render_template('index.html', articles=['dummy 1', 'dummy 2'])


@bp.route('/<foo>')
def sample(foo):
    return foo