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

## Running UCR experiment

```$ pipenv run python ./src/full.py```

or:

```$ pipenv shell```

```$ python ./src/full.py```

## Running timing test experiment

```$ pipenv run python ./src/full.py timing```

or:

```$ pipenv shell```

```$ python ./src/full.py timing```


## Generating the heatmaps and box plot.


```$ pipenv run python ./src/full.py plots```

or:

```$ pipenv shell```

```$ python ./src/full.py plots```

## Monitoring progress

Both tests take a long time to run.  A long, long time.  To monitor progress while running:

```$ tail -f log.txt```


