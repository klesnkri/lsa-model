#!/usr/bin/env python
# coding: utf-8

import requests


import json


import pandas as pd


from lsa import *


def parse_data(limit=100):
    articles = []
    with open('recipes.json') as file:
        for i, line in enumerate(file):
            if i == limit:
                break
                
            obj = json.loads(line)

            text = '{}\n'.format(obj['Description'])
            for ing in obj['Ingredients']:
                text += ing + '\n'
            for met in obj['Method']:
                text += met + '\n'
            articles.append({
                'title': obj['Name'],
                'author': obj['Author'],
                'link': obj['url'],
                'text': text,
            })
    return pd.DataFrame(articles)


df = parse_data()


def save_data(df):
    df.to_csv('recipes.csv', index=False)


df


save_data(df)


article_data = load_data('articles.csv')


food_data = load_data('recipes.csv')


article_data


food_data


data = pd.concat([article_data, food_data], ignore_index=True)

