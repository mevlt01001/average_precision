from box import TruthBox, PredBox
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class Image:
    def __init__(self):
        self.truth_boxes: Optional[list[TruthBox]] = []
        self.pred_boxes: Optional[list[PredBox]] = []
        self.CM_data: list[list[int]] = [] # confidence_score, TP, FP, FN

    def load_truth_boxes(self, truth_boxes: list):
        for truth_box in truth_boxes:
            new_truth_box = TruthBox(truth_box[0], truth_box[1], truth_box[2], truth_box[3], truth_box[4])
            self.truth_boxes.append(new_truth_box)
    
    def load_pred_boxes(self, pred_boxes: list):
        for pred_box in pred_boxes:
            new_pred_box = PredBox(pred_box[0], pred_box[1], pred_box[2], pred_box[3], pred_box[4], pred_box[5])
            self.pred_boxes.append(new_pred_box)

    def load_boxes(self, truth_boxes: list, pred_boxes: list, iou_treshold = 0.5):
        self.load_truth_boxes(truth_boxes)
        self.load_pred_boxes(pred_boxes)
        self.compute_confusion_matrix(iou=iou_treshold)

    def match_boxes(self, iou: float):
        for truth_box in self.truth_boxes:
            if truth_box.matched_box is not None:
                continue

            best_pred_box = None
            best_iou = -1

            for pred_box in self.pred_boxes:
                if pred_box.matched_box is not None:
                    continue

                current_iou = truth_box.iou(pred_box)
                if current_iou > best_iou and pred_box.class_id == truth_box.class_id:
                    best_iou = current_iou
                    best_pred_box = pred_box
                    

            if best_pred_box is not None and best_iou >= iou:
                truth_box.matched_box = best_pred_box
                best_pred_box.matched_box = truth_box
                best_pred_box.iou_val = best_iou


    def compute_confusion_matrix(self, iou: float):
        self.match_boxes(iou)

        for truth_box in self.truth_boxes: 
            if truth_box.matched_box is None: #fn
                self.CM_data.append([0, 0, 0, 1])
            else:# tp
                self.CM_data.append([truth_box.matched_box.conf, 1, 0, 0])

        for pred_box in self.pred_boxes:
            if pred_box.matched_box is None:
                self.CM_data.append([pred_box.conf, 0, 1, 0])

    def get_cm(self):
        return self.CM_data


class Frames:
    def __init__(self, whole_truth_boxes, whole_pred_boxes, iou_treshold=0.5):
        self.whole_truth_boxes = whole_truth_boxes
        self.whole_pred_boxes = whole_pred_boxes
        self.images: list[Image] = []
        self.data = []
        self.load_images(iou_treshold)

    def load_images(self, iou_treshold = 0.5):
        for i in range(len(self.whole_truth_boxes)):
            print(f"{i+1}/{len(self.whole_truth_boxes)}")
            image = Image()
            image.load_boxes(self.whole_truth_boxes[i], self.whole_pred_boxes[i], iou_treshold=iou_treshold)
            self.images.append(image)
            self.data.extend(image.get_cm())#conf, TP, FP, FN
        os.system('cls')


        self.data = np.array(self.data)
        self.data = self.data[np.argsort(self.data[:, 0])[::-1]]

        cum_data = self.data[:, 1:].cumsum(axis=0)#cum_TP, cum_FP, cum_FN
        precision = (cum_data[:, 0] / (cum_data[:, 0] + cum_data[:, 1])).reshape(-1,1)
        recall = (cum_data[:, 0] / sum([len(truth_boxes) for truth_boxes in self.whole_truth_boxes])).reshape(-1,1)
        f1_score = self.calc_f1_score(precision, recall).reshape(-1, 1)

        self.data = np.concatenate((self.data, cum_data, precision, recall, f1_score), axis=1)
        self.data = pd.DataFrame(self.data, columns=['conf', 'TP', 'FP', 'FN', 'cum_TP', 'cum_FP', 'cum_FN', 'precision', 'recall', 'f1_score'])
        self.data.to_csv("data.csv", index=False)      

    def calc_f1_score(self, precision, recall):
        return 2 * (precision * recall) / (precision + recall)
    
    def plot_ap(self):
        recall = self.data['recall'].to_numpy()
        precision = self.data['precision'].to_numpy()

        plt.plot(recall, precision, label=f"AP: {self.calc_mAP():.4f}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('precision_recall_curve.png')
        plt.show()
    
    def calc_mAP(self):
        recall = self.data['recall'].to_numpy()
        precision = self.data['precision'].to_numpy()
        average_precision = np.trapezoid(y=precision, x=recall, dx=0.0001)

        return average_precision
