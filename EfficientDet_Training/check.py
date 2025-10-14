import json
ann_path = "../Near_Drowning_Repository/Yet-Another-EfficientDet-Pytorch/datasets/Near_Drowning_Detector-8/annotations/fixed_instances_train2017.json"
js = json.load(open(ann_path))
cats = sorted(js['categories'], key=lambda x: x['id'])
print('in train : ')
print([c['name'] for c in cats])

ann_path = "../Near_Drowning_Repository/Yet-Another-EfficientDet-Pytorch/datasets/Near_Drowning_Detector-8/annotations/fixed_instances_val2017.json"
js = json.load(open(ann_path))
cats = sorted(js['categories'], key=lambda x: x['id'])
print('in valid : ')
print([c['name'] for c in cats])
