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

        # TP = truth_box.matched_box is not None and truth_box.class_id == truth_box.matched_box.class_id | A truth box has matched a pred box that has same class id with it
        # TN = İgnore
        # FP = pred_box.matched_box == None or pred_box.matched_box.class_id != pred_box.class_id | A pred box hasn't matched any truth box or it has matched a truth box that hasn't same class id with it
        # FN = truth_box.matched_box == None

        for truth_box in self.truth_boxes: 
            if truth_box.matched_box is None:# if a truth box hasn't matched a predicted box
                self.confusion_matrix.FN += 1
            else:# if a truth box has matched a predicted box
                if truth_box.class_id == truth_box.matched_box.class_id: # if a truth box has matched a predicted box and it's class id has same value with it's predicted box's class id
                    self.confusion_matrix.TP += 1
                else:# if a truth box has matched a predicted box and it's class id has not same value with it's predicted box's class id
                    self.confusion_matrix.FP += 1

        for pred_box in self.pred_boxes:
            if pred_box.matched_box is None:# if a pred box hasn't match a truth box
                self.confusion_matrix.FP += 1



