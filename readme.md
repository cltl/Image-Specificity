# Image specificity

This repository contains everything to generate image specificity scores for a set of images, based on their descriptions.
More details are provided in our COLING paper. If you've found this code useful, please cite:

```
@inproceedings{miltenburg2018DIDEC,
	Author = {Emiel van Miltenburg and \'Akos K\'adar and Ruud Koolen and Emiel Krahmer},
	Booktitle = {Proceedings of COLING},
	Publisher = {ACL},
	Title = {DIDEC: The Dutch Image Description and Eye-tracking Corpus},
	Year = {2018}}
```

## Requirements
We used Python 3.6.1, to optimize the specificity metric using the `@lru_cache` decorator.
This decorator is not implemented in Python 2.7.

* The Natural Language Toolkit (NLTK, we used 3.2.2), with the WordNet corpus installed.
* Numpy (we used 1.12.0)
* Scipy (we used 0.18.1)
* Scikit-learn (we used 0.18.1)
* Tabulate (We used 0.7.7)
* Gensim (We used 2.3.0)

## Instructions
* Download the original MS COCO captions from [here](http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip).
* Unpack `captions_train-val2014`, and rename the folder to `coco-annotations`.
* Run `python prepare_data.py` to create `to_process.json` from `imgs2.tsv`.
* Run `python generate_coco_scores.py` to generate `coco_specificity_scores.csv`.
* Run `analyze_scores.py` to produce `distributions.pdf`, and to print the statistics.

## Details on computing specificity
The original image specificity script (`calculate_automated_specificity.py`) was downloaded from [here](https://github.com/jasmainak/specificity).
We split the specificity computation into separate functions, and added a cache for the word to word similarity.
This bypasses the costly procedure to compute the WordNet distance the synsets of the two words.

## Scripts

The two **main scripts** are:

* `reimplementation.py` provides our reimplementation of the image specificity metric.
* `vector_similarity.py` provides a vector-based alternative.

We provide two scripts that check whether the implemented metrics perform well.
Modify the `VECTOR_LOCATION` variable to the right path for your embeddings.

* `test_reimplementation.py` checks whether our reimplementation produces the same scores as the original script.
* `test_vectors.py` checks how well the vector-based replacement performs.

Other scripts are used to generate scores for MS COCO.

* `prepare_data.py` prepares the data.
* `generate_coco_scores.py` generates the scores.
* `analyze_scores.py` analyzes the resulting scores.
