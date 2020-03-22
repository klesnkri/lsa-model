from shutil import copyfile

def enable():
    copyfile('jupyter_autosave_hook', '.git/hooks/pre-commit')
    print('Jupyter autosave enabled')

if __name__ == "__main__":
    enable()