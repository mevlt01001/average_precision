from box import PredBox,TruthBox,Box
import numpy as np
import matplotlib.pyplot as plt

class AP:
    truth_boxes : list[TruthBox]
    pred_boxes : list[PredBox]
    __conf_TP_FP_data : np.array
    __cumulative_TP_FP_data : np.array
    precision_recall: np.array
    AP: float

    def __init__(self, truth_boxes: list[list], pred_boxes: list[list], iou_treshold = 0.5):
        """
        `truth_boxes` is a list of lists of the form [x1, y1, x2, y2]\n
        `pred_boxes` is a list of lists of the form [x1, y1, x2, y2, conf]
        """
        self.truth_boxes = AP.__load_truth_boxes(truth_boxes)
        self.pred_boxes = AP.__load_pred_boxes(pred_boxes)
        AP.__match_boxes(self.truth_boxes, self.pred_boxes, iou_treshold)
        self.__conf_TP_FP_data = AP.__compute_TP_FP(self.pred_boxes)
        sorted_conf_TP_FP_data = self.__conf_TP_FP_data[self.__conf_TP_FP_data[:,0].argsort()][::-1]
        self.__cumulative_TP_FP_data = np.cumsum(sorted_conf_TP_FP_data[:,1:], axis=0)
        self.precision_recall = AP.__calc_Precision_Recall(self.__cumulative_TP_FP_data, len(self.truth_boxes))
        self.mAP = AP.__calc_AP(self.precision_recall)

    def __load_truth_boxes(truth_boxes: list):
        """
        `truth_boxes` is a list of lists of the form [x1, y1, x2, y2]
        """
        _truth_boxes = []
        for truth_box in truth_boxes:
            x1,y1,x2,y2 = truth_box
            new_truth_box = TruthBox(x1,y1,x2,y2)
            _truth_boxes.append(new_truth_box)

        return _truth_boxes

    def __load_pred_boxes(pred_boxes: list):
        """
        `pred_boxes` is a list of lists of the form [x1, y1, x2, y2, conf]
        """
        _pred_boxes = []
        for pred_box in pred_boxes:
            x1,y1,x2,y2,conf = pred_box
            new_pred_box = PredBox(x1,y1,x2,y2,conf)
            _pred_boxes.append(new_pred_box)

        return _pred_boxes

    
    def __match_boxes(truth_boxes: list[TruthBox], pred_boxes: list[PredBox], iou_threshold: float):

        pred_boxes.sort(key=lambda p: p.conf, reverse=True)

        for pred_box in pred_boxes:
            if pred_box.matched_box is None:
                best_iou = 0
                best_truth_box = None
                for truth_box in truth_boxes:
                    if truth_box.matched_box is None:
                        current_iou = Box.iou(pred_box, truth_box)
                        if current_iou > best_iou:
                            best_iou = current_iou
                            best_truth_box = truth_box
                if best_iou > iou_threshold:
                    pred_box.match(best_truth_box)

    def __compute_TP_FP(pred_boxes: list[PredBox]):

        TP_FP_data = []
        for pred_box in pred_boxes: 
            if pred_box.matched_box is not None:
                TP_FP_data.append([pred_box.conf,1,0]) # Eşleşme yapılırsa TP
            else:
                TP_FP_data.append([pred_box.conf,0,1]) # Eşleşme yapılmazsa FP

        return np.array(TP_FP_data)
    
    def __calc_Precision_Recall(cumulative_TP_FP_data: np.array, total_truth_boxes_count: int):
        precision = (cumulative_TP_FP_data[:, 0] / ((cumulative_TP_FP_data[:, 0] + cumulative_TP_FP_data[:, 1]))).reshape(-1,1)
        recall = (cumulative_TP_FP_data[:, 0] / total_truth_boxes_count).reshape(-1,1)
        return np.concatenate((precision, recall), axis=1)
    
    def __calc_AP(precision_recall: np.array):
        return np.trapz(y=precision_recall[:,0], x=precision_recall[:,1], dx=0.0001)
    
    def plot_precision_recall(self):
        plt.plot(self.precision_recall[:,1], self.precision_recall[:,0])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.show()
