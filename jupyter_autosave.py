from shutil import copyfile
from os import remove

def enable():
    copyfile('jupyter_autosave_hook', '.git/hooks/pre-commit')
    print('Jupyter autosave enabled')

def disable():
    remove('.git/hooks/pre-commit')
    print('Jupyter autosave enabled')

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