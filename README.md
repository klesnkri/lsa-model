# LSA model
Implementation of Latent semantic analysis model. Functionality visualized on finding similar articles in a database. Demo available on [https://bi-vwm-lsi-demo.herokuapp.com/0](https://bi-vwm-lsi-demo.herokuapp.com/).

## Installation

```
$ ./init.sh
```

Install packages as needed on errors and rerun init.

## Getting started

### Running the server

```
$ ./run.sh # displays address and port of server (e.g. localhost:5000)
```

### Modifying server data

1. replace data inside `server/data/`
2. update source code to reflect changes
3. recompute LSA - `$ flask update`

### Modifying LSA parameters

0. `$ export FLASK_APP=server` # if not already set
1. modify values in `server/lsa_config.json`
2. recompute LSA - `$ flask update`

### Initializing/Recomputing LSA 

```
$ flask update # may take some time
```

## Project structure

```
├── lsa                # LSA package
│   ├── data                     # selected usable data
│   ├── raw_data                 # unprocessed/large data - not saved to git
│   ├── data.ipynb               # notebook for data exploration
│   ├── lsa.py                   # LSA related code
│   └── select_articles.py       # script to select articles
├── server             # demo server
│   ├── static
│   ├── templates
│   ├── cache                    # LSA cache files used by server
│   ├── data                     # data files used by server
│   ├── __init__.py              # server + CLI initialization
│   ├── lsa_config.py            # config file with LSA parameters
│   └── views.py                 # server request handling
├── articles_source.txt          # data sources
├── init.sh                      # install this package
├── README.md
├── run.sh                       # start server
├── setup.py                     # package setup file
└── .gitignore
```
## About
Created as a semestral project for the [Searching Web and Multimedia Databases](http://bilakniha.cvut.cz/cs/predmet1449106.html) course at [FIT CTU](https://fit.cvut.cz/).

## Authors
David Mašek and Kristýna Klesnilová
