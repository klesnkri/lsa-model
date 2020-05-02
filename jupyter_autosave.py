from shutil import copyfile
from os import remove
from os.path import isfile

def enable():
    copyfile('jupyter_autosave_hook', '.git/hooks/pre-commit')
    print('Jupyter autosave enabled')

def disable():
    if isfile('.git/hooks/pre-commit'):
        remove('.git/hooks/pre-commit')
        print('Jupyter autosave disabled')
    else:
        print('Already disabled or error :/')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['enable', 'disable'])
    args = parser.parse_args()
    if args.action == 'enable':
        enable()    
    elif args.action == 'disable':
        disable()
    else:
        raise ValueError('invalid action')