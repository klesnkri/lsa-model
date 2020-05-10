#!/bin/bash
export FLASK_APP=server
python nltk_build.py
flask update
