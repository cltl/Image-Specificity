from scipy.stats import spearmanr
from reimplementation import analyze_corpus, image_specificity
import csv
import json

def load_data():
    "Load the data to process."
    with open('./to_process.json') as f:
        data = json.load(f)
        ids, images = zip(*data['coco_data'].items())
        selection = set(data['selection'])
    return ids, images, selection


def get_scores(images):
    """
    Get the image specificity scores for a list of images.
    Images correspond to lists of descriptions.
    """
    # Preprocess the corpus
    vectorizer, analyzer = analyze_corpus(images)
    scores = []
    # Compute the image specificity scores.
    for i, image in enumerate(images):
        if i % 10 == 0:
            print(i)
        score = image_specificity(image, vectorizer, analyzer)
        scores.append(score)
    return scores


def get_rows(ids, scores, selection):
    "Row generator to format the data for output as a csv file."
    for flickr_id, specificity_score in zip(ids, scores):
        selected = flickr_id in selection
        yield [flickr_id, selected, specificity_score]


ids, images, selection = load_data()
scores = get_scores(images)
rows = get_rows(ids, scores, selection)

with open('coco_specificity_scores.csv','w') as f:
    writer = csv.writer(f)
    writer.writerow(['flickr_id', 'selected', 'specificity'])
    writer.writerows(rows)
