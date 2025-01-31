from box import TruthBox, PredBox
from confusion_matrix import ConfusionMatrix
from typing import Optional

class Image:
    def __init__(self):
        self.truth_boxes: Optional[list[TruthBox]] = None
        self.pred_boxes: Optional[list[PredBox]] = None
        self.confusion_matrix = ConfusionMatrix()
        

    def load_truth_boxes(self, truth_boxes: list[TruthBox]):
        self.truth_boxes = truth_boxes
    
    def load_pred_boxes(self, pred_boxes: list[PredBox]):
        self.pred_boxes = pred_boxes

    def load_boxes(self, truth_boxes: list[TruthBox], pred_boxes: list[PredBox]):
        self.truth_boxes = truth_boxes
        self.pred_boxes = pred_boxes

    def match_boxes(self, iou: float):
        for truth_box in self.truth_boxes:
            
            if truth_box.matched_box is not None:
                continue
            
            best_pred_box: PredBox = None
            best_iou = -1

            for pred_box in self.pred_boxes:
                if pred_box.matched_box is not None:
                    continue

                current_iou = truth_box.iou(pred_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_pred_box = pred_box
            
            if best_pred_box is not None and best_iou > iou:
                truth_box.matched_box = best_pred_box
                best_pred_box.matched_box = truth_box
                best_pred_box.iou = best_iou                



    def compute_confusion_matrix(self, iou: float):
        self.match_boxes(iou)

        for 

