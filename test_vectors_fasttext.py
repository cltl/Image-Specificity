from scipy import nanmean
from scipy.io import loadmat
from scipy.stats import spearmanr

from vector_similarity import analyze_corpus, image_specificity
from gensim.models.wrappers import FastText

VECTOR_LOCATION = '/Path/To/Vectors/wiki.en/wiki.en.bin'

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
model = FastText.load_fasttext_format(VECTOR_LOCATION)
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

# Original correlation from Jas & Parikh:    SpearmanrResult(correlation=0.68934200875390916, pvalue=3.5414150507978258e-126)
# Correlation with the original ratings:     SpearmanrResult(correlation=0.68927476909024299, pvalue=3.8299986828805149e-126)
# Correlation with the original estimations: SpearmanrResult(correlation=0.86427626648134603, pvalue=1.2412261841386086e-266)
