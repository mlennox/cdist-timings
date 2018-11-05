# benchmarking cosine similarity implementations

Benchmarking scipy.spatial.distance.cdist implementations.

As part of creating an encoder/decoder model, we want to find the `cosine` distance between each row in one array and each row in a second. One implementation will loop through the first array row by row, find the cosine distance and keep track of the lowest value. The second will take a vectorised approach to find the cosine similarity between the two matrices and then find the smallest cosine distance for each row.

## pre-requisites

- Python 3.6.3 / pip 18.0 (I use Pyenv and virutalenv)

### Preparing the environment

The `setup.sh` script may be helpful in setting up your environment, assuming you have already installed `pyenv` and `virtualenv` (see my tutorial [Python dependency - hell no!](http://www.webpusher.ie/2018/09/19/python-dependency-hell-no/))

As you need to use Seaborn, which requires `matplotlib` you will need to have installed Python 3.6.3 as a framework, using Pyenv. This is relatively easy, [see my tutorial on installing](https://www.webpusher.ie/2018/07/07/pyenv-matplotlib-high-sierra/)

The script contains the following

```bash
pyenv virtualenv 3.6.3 cdisttimings
pyenv local cdisttimings

pip install -U pip

pip install -r requirements.txt
```

Once you run that you should be ready to go.

## Why cosine distance? Why not euclidian distance?

The use-case here is finding similarities between the word embeddings in two matrices - for word embeddings the amplitude of the vector does not matter as they should have been normalised, the angle between vectors is important.

The Euclidian distance finds the geometrical distance between the vectors, while the cosine distance (or similarity) finds the angle between the vectors.

![](./Figure1.png)
