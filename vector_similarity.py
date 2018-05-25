from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
from functools import lru_cache
from numpy import average
from scipy import nanmean


def py2max(iterable):
    "Version of max that is compatible with Python 2."
    placeholder = [i for i in iterable if not i == None]
    if len(placeholder) == 0:
        return None
    else:
        return max(placeholder)


def analyze_corpus(images):
    "Preprocess the corpus."
    vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w\\w\\w+\\b')
    documents = [' '.join(descriptions) for descriptions in images]
    vectorizer.fit(documents)
    analyzer = vectorizer.build_analyzer()
    return vectorizer, analyzer


@lru_cache(maxsize=1000000)
def w2w(word1, word2, model):
    "Get the similarity between two words."
    try:
        return model.similarity(word1,word2)
    except KeyError:
        return 0

def word2word(word1, word2, model):
    "Sort the words and then call the actual function."
    a,b = sorted([word1, word2])
    return w2w(a,b, model)


def word2sent(word, sent, model):
    "Get best similarity score between word and all words in a sentence."
    similarities = [word2word(word, ref, model) for ref in sent]
    return py2max(similarities)


def vector_similarities(words1, words2, model):
    "Get wordnet similarities between two tokenized sentences."
    sims1 = [word2sent(word, words2, model) for word in words1]
    sims2 = [word2sent(word, words1, model) for word in words2]
    return sims1, sims2


def sentence_similarity(sent1, sent2, vectorizer, analyzer, model):
    "Get the similarity between two sentences."
    words1 = analyzer(sent1)
    words2 = analyzer(sent2)
    # Compute weights
    sent1_weights = [vectorizer.transform([sent1]).toarray()[0][vectorizer.vocabulary_.get(w)] for w in words1]
    sent2_weights = [vectorizer.transform([sent2]).toarray()[0][vectorizer.vocabulary_.get(w)] for w in words2]
    # Compute sentence similarity
    sims1, sims2 = vector_similarities(words1, words2, model)
    # Concatenate the results.
    combined_sims = sims1 + sims2
    combined_weights = sent1_weights + sent2_weights
    if all(x is None for x in combined_sims):
        return float('nan')
    else:
        (sim_cleaned, a) = zip(*[(x, w) for (x, w) in zip(combined_sims, combined_weights)
                                        if not x is None])
        similarity = average(sim_cleaned, weights=a)
        return similarity


def image_specificity(descriptions, vectorizer, analyzer, model):
    "Compute image specificity."
    similarities = [sentence_similarity(sent1, sent2, vectorizer, analyzer, model)
                    for sent1, sent2 in combinations(descriptions, 2)]
    specificity = nanmean(similarities)
    return specificity
