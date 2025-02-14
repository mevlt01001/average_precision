from box import TruthBox, PredBox
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class Image:
    def __init__(self):
        self.truth_boxes: Optional[list[TruthBox]] = [] # x1, y1, x2, y2, class_id
        self.pred_boxes: Optional[list[PredBox]] = [] # x1, y1, x2, y2, class_id, conf
        self.boxes_data: list[list] = [] # confidence_score, TP, FP, FN
        self.TP = 0
        self.FP = 0
        self.FN = 0


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
                self.boxes_data.append([0, 0, 0, 1])
                self.FN += 1
            else:# tp
                self.boxes_data.append([truth_box.matched_box.conf, 1, 0, 0])
                self.TP += 1

        for pred_box in self.pred_boxes:
            if pred_box.matched_box is None:
                self.boxes_data.append([pred_box.conf, 0, 1, 0])
                self.FP += 1

        self.precision = self.TP / ((self.TP + self.FP)+0.000000001)
        self.recall = self.TP / ((self.TP + self.FN)+0.000000001)
        self.CM_data = [self.precision, self.recall]


class Frames:
    def __init__(self, whole_truth_boxes, whole_pred_boxes, iou_treshold=0.5):
        self.whole_truth_boxes: list = whole_truth_boxes
        self.whole_pred_boxes: list = whole_pred_boxes
        self.images: list[Image] = []
        self.data = []
        self.data_per_image = []
        self.load_images(iou_treshold)

    def load_images(self, iou_treshold = 0.5):
        for i in range(len(self.whole_truth_boxes)):
            #print(f"{i+1}/{len(self.whole_truth_boxes)}")
            image = Image()
            image.load_boxes(self.whole_truth_boxes[i], self.whole_pred_boxes[i], iou_treshold=iou_treshold)
            self.images.append(image)
            self.data_per_image.append(image.CM_data)
            self.data.extend(image.boxes_data)#conf, TP, FP, FN
        #os.system(f"{'cls' if os.name == 'nt' else 'clear'}")

        self.data = np.array(self.data)
        self.data_per_image = np.array(self.data_per_image)
        avg_precision = self.data_per_image[:, 0].mean()
        avg_recall = self.data_per_image[:, 1].mean()
        self.data = self.data[np.argsort(self.data[:, 0])[::-1]]

        cum_data = self.data[:, 1:].cumsum(axis=0)#cum_TP, cum_FP, cum_FN
        precision = (cum_data[:, 0] / ((cum_data[:, 0] + cum_data[:, 1]))).reshape(-1,1) if (cum_data[:, 0] + cum_data[:, 1]).any() != 0 else np.nan
        recall = (cum_data[:, 0] / sum([len(truth_boxes) for truth_boxes in self.whole_truth_boxes])).reshape(-1,1)
        

        self.data = np.concatenate((self.data, cum_data, precision, recall), axis=1)
        self.data = pd.DataFrame(self.data, columns=['conf', 'TP', 'FP', 'FN', 'cum_TP', 'cum_FP', 'cum_FN', 'precision', 'recall'])
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"TP: {self.data['TP'].sum()}")
        print(f"FP: {self.data['FP'].sum()}")
        print(f"FN: {self.data['FN'].sum()}")
        #self.data.to_csv(f"data_iou_{iou_treshold}.csv", index=False)  


    
    def plot_ap(self, title):
        recall = self.data['recall'].to_numpy()
        precision = self.data['precision'].to_numpy()
        
        plt.figure(figsize=(8, 5))
        plt.plot(recall, precision, color='darkblue')
        plt.fill_between(recall, precision, alpha=0.5, color='yellow', label=f'Area Under Curve: {self.calc_mAP():.5f}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True, color='red', linestyle='dotted', linewidth=0.5)
        plt.tight_layout()
        plt.show()
    
    def calc_mAP(self):
        recall = self.data['recall'].to_numpy()
        precision = self.data['precision'].to_numpy()
        average_precision = np.trapezoid(y=precision, x=recall, dx=0.0001)

        return average_precision
