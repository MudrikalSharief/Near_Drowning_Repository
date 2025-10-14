import json

# Define the input and output file paths
input_file = '../Near_Drowning_Repository/Yet-Another-EfficientDet-Pytorch/datasets/Near_Drowning_Detector-8/annotations/instances_val2017.json'
output_file = '../Near_Drowning_Repository/Yet-Another-EfficientDet-Pytorch/datasets/Near_Drowning_Detector-8/annotations/fixed_instances_val2017.json'

# Define the category names and their IDs
old_class_name = 'near-drowning'
new_class_name = 'near-drowning'

# Define the old ID that needs to be replaced
old_id = 0

# 1. Read the JSON file
with open(input_file, 'r') as f:
    data = json.load(f)

# 2. Get the new ID for the correct 'near-drowning' class
new_id = None
for cat in data['categories']:
    # Find the correct 'near-drowning' entry (which is not the old ID 0)
    if cat['name'] == new_class_name and cat['id'] != old_id:
        new_id = cat['id']
        break

if new_id is None:
    raise ValueError(f"Could not find a valid ID for '{new_class_name}' in the categories.")

# 3. Update annotations
print("Updating annotations...")
for ann in data['annotations']:
    if ann['category_id'] == old_id:
        ann['category_id'] = new_id

# 4. Remove the duplicate category
print("Removing duplicate category...")
data['categories'] = [cat for cat in data['categories'] if cat['id'] != old_id]

# 5. Save the fixed JSON file
print("Saving fixed data...")
with open(output_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Fix complete! Corrected file saved as {output_file}")