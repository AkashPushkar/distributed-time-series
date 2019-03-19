# dsitributed-time-series
This repository is for 10605 class project at CMU 

## Dependencies

`Pipenv` - a python virtual environment and dependencies manager. Installation on OSX via `homebrew`: 

```$ brew install pipenv```

## Setup

The `setup.sh` script will unzip the dataset archive and install the python dependencies

```$ ./setup.sh```

## Running the simple tsfresh test

```$ pipenv run python ./src/test.py```

or:

```$ pipenv shell```

```$ python ./src/test.py```

## Running the full tsfresh experiment

```$ pipenv run python ./src/full.py```

or:

```$ pipenv shell```

```$ python ./src/full.py```

