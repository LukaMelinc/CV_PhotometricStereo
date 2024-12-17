#!/bin/bash

# create, activate virtual env
echo "Creating a virtual enviroment"
python3 -m venv venv
source venv/bin/activate

# depenencies
echo "Installing dependencies"
pip install --upgrade pip
pip install -r requirements.txt

# running the script
echo "Running the Python script"
python3 main.py