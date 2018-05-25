from scipy import nanmean
from scipy.io import loadmat
from scipy.stats import spearmanr

from vector_similarity import analyze_corpus, image_specificity
from gensim.models.keyedvectors import KeyedVectors

VECTOR_LOCATION = "/Path/To/GoogleNews/GoogleNews-vectors-negative300.bin.gz"

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

print("Loading vectors.")
model = KeyedVectors.load_word2vec_format(VECTOR_LOCATION, binary=True)
print("Loaded.")

automatic, ratings = original_ratings()
images = load_images()

vectorizer, analyzer = analyze_corpus(images)
scores = []
for i, image in enumerate(images):
    if i % 10 == 0:
        print(i)
    score = image_specificity(image, vectorizer, analyzer, model)
    scores.append(score)


result = spearmanr(automatic, ratings)
print('Original correlation from Jas & Parikh:   ', result)

result = spearmanr(scores, ratings)
print('Correlation with the original ratings:    ', result)

result = spearmanr(scores, automatic)
print('Correlation with the original estimations:', result)
