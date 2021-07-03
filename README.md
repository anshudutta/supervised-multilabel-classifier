# Supervised Multi-Label Classifier

Supervised Multi-Label Classifier is a Python library for multi-class multi-label classification. This algorithm can classify texts into pre trained categories.

An example is provided on classifying movie genre based on plot synopsys

## Prerequisites

- Install pre-trained Word2Vec [model](https://code.google.com/archive/p/word2vec/) on GoogleNews, unzip and place it in the `/models` folder.

```bash
 ./scripts/word2vec300.sh
```

- Install reuters corpus from nltk to run the test

```python
import nltk
nltk.download('reuters')

```

## Installation

Use the requirements.txt file to install dependencies

```bash
python3 -m venv ./venv
source ./venv/bin/activate  
pip install -r requirements.txt
```

## How to run tests

Tests are run using `pytest`. A simple model test is provided as an example with the csv file in` /data_set` folder.
A more robust test is provided to test the model with `reuters_21578` corpus

From the root folder run

```python
pytest -s -r a
```

## Usage

From the root folder run

Default mode - Reuters corpus

```python
python run.py
```

Pass csv as arguments

```python
python run.py --model "absolute/Path/To/File.csv"
```

To run the movie-matching algorithm use `data/imdb_dataset.csv`

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
