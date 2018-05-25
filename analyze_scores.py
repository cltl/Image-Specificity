from matplotlib import pyplot as plt
import pandas as pd
from scipy import std, mean, median
from tabulate import tabulate

df = pd.read_csv('coco_specificity_scores.csv')
selected = df[df['selected']==True]

ax = df['specificity'].plot.kde(label='Full dataset', legend=True)
ax = selected['specificity'].plot.kde(ax=ax, label='Selection',legend=True)

plt.savefig('distributions.pdf')

def stats(dataframe):
    scores = list(dataframe['specificity'])
    return len(scores), mean(scores), median(scores), std(scores), min(scores), max(scores)

labels = ['Len', 'Mean', 'Median', 'Std', 'Min', 'Max']
rows = zip(labels, stats(df), stats(selected))
table = tabulate(rows, headers=['Statistic', 'Full', 'Selection'])
print(table)
