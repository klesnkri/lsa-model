# LSA demo

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

1. replace data in `server/data`
2. update source code to reflect changes
3. recompute LSA (see below)

### Modifying LSA parameters

1. modify parameters
   - either in `server/__init__.py` in function `update_lsa` (preferred)
   - or directly in code
2. recompute LSA (see below)

### Recomputing LSA 

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
│   ├── images
│   ├── cache                    # LSA cache files used by server
│   ├── data                     # data files used by server
│   ├── __init__.py              # server initialization + command line interface
│   └── views.py                 # server request handling
├── articles_source.txt          # data sources
├── init.sh                      # install this package
├── README.md
├── run.sh                       # start server
├── setup.py                     # package setup file
└── .gitignore
```
