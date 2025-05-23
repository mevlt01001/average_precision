from box import PredBox,TruthBox,Box
import numpy as np, os
import matplotlib.pyplot as plt

class AP:
    truth_boxes : list[TruthBox]
    pred_boxes : list[PredBox]
    total_TP: int
    total_FP: int
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
        __conf_TP_FP_FN_data = AP.__compute_TP_FP(self.truth_boxes, self.pred_boxes)
        sorted_conf_TP_FP_FN_data = __conf_TP_FP_FN_data[__conf_TP_FP_FN_data[:,0].argsort()][::-1]
        __cumulative_TP_FP_FN_data = np.cumsum(sorted_conf_TP_FP_FN_data[:,1:], axis=0)
        self.total_TP = __cumulative_TP_FP_FN_data[-1,0]
        self.total_FP = __cumulative_TP_FP_FN_data[-1,1]
        self.total_FN = __cumulative_TP_FP_FN_data[-1,2]
        self.precision_recall = AP.__calc_Precision_Recall(__cumulative_TP_FP_FN_data, len(self.truth_boxes))
        self.AP = AP.__calc_AP(self.precision_recall)

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
        len_preds = len(pred_boxes)

        for i,pred_box in enumerate(pred_boxes):
            print(i,"/",len_preds)
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

    def __compute_TP_FP(truth_boxes: list[TruthBox], pred_boxes: list[PredBox]):

        TP_FP_FN_data = []
        for pred_box in pred_boxes: 
            if pred_box.matched_box is not None:
                TP_FP_FN_data.append([pred_box.conf,1,0,0]) # Eşleşme yapılırsa TP
            else:
                TP_FP_FN_data.append([pred_box.conf,0,1,0]) # Eşleşme yapılmazsa FP

        for truth_box in truth_boxes:
            if truth_box.matched_box is None:
                TP_FP_FN_data.append([0,0,0,1]) # Truth box'dan eşleşme yapılmazsa FN

        return np.array(TP_FP_FN_data)
    
    def __calc_Precision_Recall(cumulative_TP_FP_data: np.array, total_truth_boxes_count: int):
        precision = (cumulative_TP_FP_data[:, 0] / ((cumulative_TP_FP_data[:, 0] + cumulative_TP_FP_data[:, 1]))).reshape(-1,1)
        recall = (cumulative_TP_FP_data[:, 0] / total_truth_boxes_count).reshape(-1,1)
        return np.concatenate((precision, recall), axis=1)
    
    def __calc_AP(precision_recall: np.array):
        return np.trapz(y=precision_recall[:,0], x=precision_recall[:,1], dx=0.0001)
    
    def plot_precision_recall(self, save=False, path=None, name="PRECISION RECALL CURVE"):
        plt.plot(self.precision_recall[:,1], self.precision_recall[:,0])
        plt.fill_between(self.precision_recall[:,1], self.precision_recall[:,0], alpha=0.2)
        plt.suptitle(name)        
        plt.legend(["AP", f"Area under the curve: {self.AP:.2f}"])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim([-0.1,1.1])
        plt.ylim([-0.1,1.1])
        if save:
            plt.savefig("AP.png")
            if path is not None:
                os.system(f'mv AP.png {path}')
        plt.show()
