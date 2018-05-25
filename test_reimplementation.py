from scipy import nanmean
from scipy.io import loadmat
from scipy.stats import spearmanr

from reimplementation import analyze_corpus, image_specificity


def original_ratings():
    "Function returning the automatic specificity scores along with the human ratings."
    data = loadmat('./original_data/specificity_automated.mat')
    automatic = data['specificity_automated'][0]
    data = loadmat('./original_data/specificity_scores_MEM5S.mat')
    ratings = data['scores']
    ratings = [nanmean([nanmean(row) for row in image]) for image in ratings]
    return automatic, ratings

def load_images():
    image_data = loadmat('./original_data/memorability_888_img_5_sent.mat')
    images = [[s[0] for s in group] for group in image_data['memorability_sentences']]
    return images

automatic, ratings = original_ratings()
images = load_images()

vectorizer, analyzer = analyze_corpus(images)
scores = []
for i, image in enumerate(images):
    if i % 10 == 0:
        print(i)
    score = image_specificity(image, vectorizer, analyzer)
    scores.append(score)

result = spearmanr(automatic, ratings)
print('Original correlation from Jas & Parikh:   ', result)

result = spearmanr(scores, ratings)
print('Correlation with the original ratings:    ', result)
#Correlation with the original ratings:     SpearmanrResult(correlation=0.68575006369295999, pvalue=2.2577679158484526e-124)

result = spearmanr(scores, automatic)
print('Correlation with the original estimations:', result)
#Correlation with the original estimations: SpearmanrResult(correlation=0.99179365734772362, pvalue=0.0)
