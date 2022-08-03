# More about git submodules: https://git-scm.com/book/en/v2/Git-Tools-Submodules
git submodule init
git submodule update

# Prepare Python stuff
python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Some weird Python init stuff 
python ./project-init.py