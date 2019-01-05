# Supervised Multi-Label Classifier

Supervised Multi-Label Classifier is a Python library for multi-class multi-label classification

## Prerequisites
1. Install Pretrained Word2Vec model on GoogleNews, unzip and place it in the /models folder.
Refer https://gist.github.com/yanaiela/cfef50380de8a5bfc8c272bb0c91d6e1
2. Install reuters corpus from nltk to run the test
```python
import nltk
nltk.download('reuters')

```

## Installation

Use the requirements.txt file to install dependencies

```bash
pip install <dependencies>
```
## How to run tests
Tests are run using pytest. A simple model test is provided as an example with the csv file in /data_set folder. A more robust test is provided to test the model with reuters_21578 corpus

From the root folder run
```python
pytest -s -r a
```

## Usage
From the root folder run

```python
python run.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)