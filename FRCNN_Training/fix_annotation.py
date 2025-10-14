import json
import os

def fix_annotation_file(input_file, output_file=None, old_class_name='near-drowning', new_class_name='near-drowning', old_id=0):
    """
    Fix duplicate category IDs in COCO annotation files
    """
    if output_file is None:
        # Create output filename with 'fixed_' prefix
        dir_name = os.path.dirname(input_file)
        base_name = os.path.basename(input_file)
        output_file = os.path.join(dir_name, f"fixed_{base_name}")
    
    print(f"Fixing annotation file: {input_file}")
    
    # 1. Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # 2. Get the new ID for the correct class
    new_id = None
    for cat in data['categories']:
        # Find the correct class entry (which is not the old ID)
        if cat['name'] == new_class_name and cat['id'] != old_id:
            new_id = cat['id']
            break

    if new_id is None:
        raise ValueError(f"Could not find a valid ID for '{new_class_name}' in the categories.")

    print(f"Replacing category ID {old_id} with {new_id}")

    # 3. Update annotations
    updated_count = 0
    for ann in data['annotations']:
        if ann['category_id'] == old_id:
            ann['category_id'] = new_id
            updated_count += 1

    print(f"Updated {updated_count} annotations")

    # 4. Remove the duplicate category
    original_cat_count = len(data['categories'])
    data['categories'] = [cat for cat in data['categories'] if cat['id'] != old_id]
    removed_categories = original_cat_count - len(data['categories'])
    print(f"Removed {removed_categories} duplicate categories")

    # 5. Save the fixed JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Fix complete! Corrected file saved as {output_file}")
    return output_file

def fix_all_annotations(dataset_dir):
    """
    Fix all annotation files in the dataset directory
    """
    # Find all annotation files
    annotation_files = []
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.json') and 'annotations' in file:
                annotation_files.append(os.path.join(root, file))
    
    if not annotation_files:
        print("No annotation files found!")
        return
    
    fixed_files = []
    for ann_file in annotation_files:
        try:
            fixed_file = fix_annotation_file(ann_file)
            fixed_files.append(fixed_file)
        except Exception as e:
            print(f"Error fixing {ann_file}: {e}")
    
    return fixed_files

if __name__ == "__main__":
    # Get the current script directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(SCRIPT_DIR, "Near_Drowning_Detector-7")
    
    # Fix all annotation files
    fixed_files = fix_all_annotations(DATASET_DIR)
    print(f"\nFixed {len(fixed_files)} annotation files:")
    for file in fixed_files:
        print(f"  - {file}")