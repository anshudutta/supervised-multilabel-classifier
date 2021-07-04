# Supervised Multi-Label Classifier

Supervised Multi-Label Classifier is a Python library for multi-class multi-label classification. This algorithm can classify texts into pre trained categories.

Consider the following scenario. We have a set of `descriptions` with two classifications `ny` and `london`

| Category   | Text                                   |
| ---------- | -------------------------------------- |
| ny         | New York is a hell of a town           |
| london     | The capital of Great Britain is London |
| ny, london | New York is great and so is London.    |

Based on my training set above the classifier would classify below tests as follows

| Text                                           | Classification |
| ---------------------------------------------- | -------------- |
| New York was originally dutch                  | ny             |
| It rains a lot in London                       | london         |
| New York and London both have the stock market | ny, london     |

## Prerequisites

```bash
 ./scripts/setup.sh
```
- Installs pre-trained Word2Vec [model](https://code.google.com/archive/p/word2vec/)
- Installs reuters corpus from nltk to run the test

## Installation

Use the requirements.txt file to install dependencies

```bash
python3 -m venv ./venv
source ./venv/bin/activate  
pip install -r requirements.txt
```

## Run tests

Tests are run using `pytest`. A simple model test is provided as an example with the csv file in` /data_set` folder.
A more robust test is provided to test the model with `reuters_21578` corpus

From the root folder run

```python
pytest -s -r a
```

## Usage

### Do it yourself
From the root folder run

- Default mode - Reuters corpus

```python
python run.py
```

- Pass csv as arguments

```python
python run.py --model "path/to/file.csv"
```

To run the movie-matching algorithm use `data/imdb_dataset.csv`

### Docker

```bash
docker build . -t supervised-multi-classifier

docker run --rm -it \
-v `pwd`/supervised_multilabel_classifier/model:/home/app/supervised_multilabel_classifier/model \
-v `pwd`/supervised_multilabel_classifier/data_set:/home/app/supervised_multilabel_classifier/data_set \
supervised-multi-classifier \
 --model /home/app/supervised_multilabel_classifier/data_set/<your-model>.csv
```
### What can it do?

Let's run the movie algorithm

```bash
docker run --rm -it \
-v `pwd`/supervised_multilabel_classifier/model:/home/app/supervised_multilabel_classifier/model \
-v `pwd`/supervised_multilabel_classifier/data_set:/home/app/supervised_multilabel_classifier/data_set \
supervised-multi-classifier \
 --model /home/app/supervised_multilabel_classifier/data_set/imdb_dataset.csv
 
 ...
 
Enter/Paste your content. Ctrl-D or Ctrl-Z ( windows ) to save it.
I kissed a girl, I liked it
prediction: ['Drama', 'Romance', 'Comedy', 'Fantasy']
Most similar: [(0.6800497770309448, 'tt0102250/L.A. Story'), (0.6765267848968506, 'tt0386005/New Police Story'), (0.6711339950561523, 'tt0338763/Battle Royale II'), (0.669927179813385, 'tt0101507/Boyz n the Hood'), (0.6669382452964783, 'tt0258068/The Quiet American')]
```

The description: `I kissed a girl, I liked it`

Prediction: `Drama, Romance, Comedy, Fantasy`

Try it with proper plot synopsys

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
