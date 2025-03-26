class Box:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.xyxy = [self.x1, self.y1, self.x2, self.y2]
        self.matched_box: Box = None

class PredBox(Box):
    def __init__(self, x1, y1, x2, y2, conf):
        super().__init__(x1, y1, x2, y2)
        self.conf = conf
        self.iou_val = 0

class TruthBox(Box):
    def __init__(self, x1, y1, x2, y2):
        super().__init__(x1, y1, x2, y2)

    def iou(self, pred_box: PredBox) -> float:
        intersect_width = max(0, min(self.x2, pred_box.x2) - max(self.x1, pred_box.x1))
        intersect_height = max(0, min(self.y2, pred_box.y2) - max(self.y1, pred_box.y1))
        if intersect_width == 0 or intersect_height == 0:
            return 0
        intersect = intersect_width * intersect_height
        truth_area = (self.x2 - self.x1) * (self.y2 - self.y1)
        pred_area = (pred_box.x2 - pred_box.x1) * (pred_box.y2 - pred_box.y1)
        union = truth_area + pred_area - intersect
        iou = intersect / union        
        return iou
    
    def match(self, pred_box: PredBox, iou: float):
        self.matched_box = pred_box
        pred_box.matched_box = self
        pred_box.iou_val = iou

        
