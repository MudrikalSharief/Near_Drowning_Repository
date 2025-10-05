import random
import torchvision.transforms.functional as F

class Resize:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, image, target):
        # Get original size
        orig_w, orig_h = image.size  # PIL image gives (w, h)
        new_h, new_w = self.size

        # Resize the image
        image = F.resize(image, [new_h, new_w])

        if "boxes" in target:
            boxes = target["boxes"]
            if boxes.numel() > 0:
                # Scale bounding boxes
                boxes = boxes.clone()
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_w / orig_w)
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_h / orig_h)
                target["boxes"] = boxes

        return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            if "boxes" in target and target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                width = image.shape[2]  # image: CxHxW
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
