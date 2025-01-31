from typing import Optional

class Box:
    def __init__(self, x1, y1, x2, y2, class_id):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.class_id = class_id
        self.matched_box: Optional[Box] = None



class PredBox(Box):
    def __init__(self, x1, y1, x2, y2, class_id):
        super().__init__(x1, y1, x2, y2, class_id)
        self.iou=0


class TruthBox(Box):
    def __init__(self, x1, y1, x2, y2, class_id):
        super().__init__(x1, y1, x2, y2, class_id)

    def iou(self, pred_box: PredBox) -> float:
        intersect = (min(self.x2, pred_box.x2) - max(self.x1, pred_box.x1)) * (min(self.y2, pred_box.y2) - max(self.y1, pred_box.y1))
        if intersect <= 0:
            return 0
        union = (self.x2 - self.x1) * (self.y2 - self.y1) + (pred_box.x2 - pred_box.x1) * (pred_box.y2 - pred_box.y1) - intersect
        return intersect / union

        
