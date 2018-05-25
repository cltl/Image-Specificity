import csv
import json
from collections import defaultdict
from itertools import chain

with open('./imgs2.tsv') as f:
    reader = csv.DictReader(f, fieldnames=['genome', 'coco_id', 'url'], delimiter='\t')
    entries = list(reader)

coco_ids = [e['coco_id'] for e in entries]

with open('./coco-annotations/captions_train2014.json') as f:
    coco_train = json.load(f)

index = defaultdict(list)
for entry in coco_train['annotations']:
    coco_id = entry['image_id']
    caption = entry['caption']
    index[coco_id].append(caption)

data = {'coco_data': index, 'selection': coco_ids}

with open('to_process.json','w') as f:
    json.dump(data, f)
