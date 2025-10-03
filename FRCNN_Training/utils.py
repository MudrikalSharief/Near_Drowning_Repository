from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt


def evaluate_coco(model, data_loader, coco_gt, device, inv_cat_id_map):
    model.eval()
    results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"])
            boxes = output["boxes"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score < 0.05:
                    continue
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                results.append({
                    "image_id": image_id,
                    "category_id": inv_cat_id_map[int(label)],
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                })

    if len(results) == 0:
        print("⚠️ No predictions to evaluate!")
        return None

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = {
        "mAP@[.5:.95]": coco_eval.stats[0],
        "mAP@0.50": coco_eval.stats[1],
        "Recall": coco_eval.stats[8],
        "Precision": coco_eval.stats[0],
    }
    return stats


def plot_training_curves(train_losses, val_losses, map50s, map5095s, precisions, recalls):
    plt.figure(figsize=(15, 5))

    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)

    # mAP curves
    plt.subplot(1, 3, 2)
    plt.plot(map50s, label="mAP@0.50")
    plt.plot(map5095s, label="mAP@[.5:.95]")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Validation mAP")
    plt.legend()
    plt.grid(True)

    # Precision & Recall curves
    plt.subplot(1, 3, 3)
    plt.plot(precisions, label="Precision")
    plt.plot(recalls, label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Precision & Recall")
    plt.legend()
    plt.grid(True)

    plt.show()
