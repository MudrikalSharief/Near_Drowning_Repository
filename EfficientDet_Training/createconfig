# 4) Create a project yaml (projects/<PROJECT_NAME>.yml)
# Save this content as Yet-Another-EfficientDet-Pytorch/projects/my_custom_project.yml
# You can create it with a cell below or with echo > file.

yaml_content = """
project_name: Near_Drowning_Detector
train_set: train2017
val_set: val2017
num_gpus: 1
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]
anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'
obj_list: ['near_drowning','outside_of_water','swimming']
"""

# Save the YAML
with open("../Near_Drowning_Repository/Yet-Another-EfficientDet-Pytorch/projects/Near_Drowning_Detector.yml", "w") as f:
    f.write(yaml_content)

print("YAML file created!")

