import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def boxes_to_detections(boxes, class_ids=None):
    """
    Convert list of [x1, y1, x2, y2] boxes into a sv.Detections object.
    
    Args:
        boxes: List of boxes [[x1,y1,x2,y2], ...]
        class_ids: Optional list/array of class IDs for each box. 
                   If None, defaults to class_id=0 for all boxes.
    """
    if len(boxes) == 0:
        return sv.Detections.empty()
    
    xyxy = np.array(boxes, dtype=np.float32)
    
    if class_ids is None:
        class_id_arr = np.zeros(len(boxes), dtype=int)
    else:
        class_id_arr = np.array(class_ids, dtype=int)
        if len(class_id_arr) != len(boxes):
            raise ValueError("Length of class_ids must match number of boxes")
    
    confidence = np.ones(len(boxes), dtype=np.float32)  # fixed confidence = 1.0
    
    return sv.Detections(
        xyxy=xyxy,
        class_id=class_id_arr,
        confidence=confidence
    )

def compute_map_supervision(pred_boxes, pred_classes, true_boxes, true_classes):
    preds = boxes_to_detections(pred_boxes, pred_classes)
    targets = boxes_to_detections(true_boxes, true_classes)
    
    metric = MeanAveragePrecision()
    result = metric.update(preds, targets).compute()

    print("mAP@50:95:", result.map50_95)
    print("mAP@50:   ", result.map50)
    print("mAP@75:   ", result.map75)
    
    return result

def draw_boxes(pred_boxes, true_boxes, image_size=(1024, 1024)):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 240)

    # Draw predicted boxes in red
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # Draw ground truth boxes in green
    for box in true_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

    ax.set_xlim(0, image_size[0])
    ax.set_ylim(image_size[1], 0)  # Flip y-axis (top-left origin)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pred_boxes = [
        [279.345, 482.761, 341.585, 543.730]
    ]
    pred_classes = [0]

    true_boxes = [
        [269.345, 500.761, 331.585, 543.730],
        [828.594, 700.222, 900.055, 800.994]
    ]
    true_classes = [0, 0]

    result = compute_map_supervision(pred_boxes, pred_classes, true_boxes, true_classes)
    draw_boxes(pred_boxes, true_boxes, image_size=(1024, 1024))
