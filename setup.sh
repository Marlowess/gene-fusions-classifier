#!/usr/bin/env bash

pip install virtualenv
virtualenv bioinfo_project
echo "Virtual environment installed correctly."

echo "Installing dependencies..."
./bioinfo_project/bin/pip install -r requirements.txt

echo "Done"

exit 0