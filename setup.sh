#!/usr/bin/env bash

pip install virtualenv
virtualenv bioinfo_env
echo "Virtual environment installed correctly."

echo "Installing dependencies..."
./bioinfo_env/bin/pip install -r requirements.txt

echo "Done"

exit 0
