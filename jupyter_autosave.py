import subprocess
file_name = 'data.ipynb'
command = 'jupyter nbconvert {} --to="python" --output-dir="generated"'.format(file_name)
subprocess.run(command, shell=True, check=True) # throws subprocess.CalledProcessError on fail
