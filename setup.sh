#!/bin/bash

# Sets up the project environment

# Unzips the data if not present
if [ ! -d "./data" ]; then
  echo "Unzipping UCR data"
  mkdir "./data"
  unzip "./archives/ucr.zip" -d "./data"
fi

# Installs the dependencies
pipenv install
