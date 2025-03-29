class Box:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.xyxy = [self.x1, self.y1, self.x2, self.y2]
        self.matched_box: Box = None

    def match(self, box: 'Box'):
        self.matched_box = box
        box.matched_box = self

    @staticmethod
    def iou(box1: 'Box', box2: 'Box'):
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        intersection_area = (x2 - x1) * (y2 - y1) if not ((x2 - x1) < 0 or (y2 - y1) < 0) else 0

        box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)

        return intersection_area / (box1_area + box2_area - intersection_area)

class PredBox(Box):
    def __init__(self, x1, y1, x2, y2, conf):
        super().__init__(x1, y1, x2, y2)
        self.conf = conf

class TruthBox(Box):
    def __init__(self, x1, y1, x2, y2):
        super().__init__(x1, y1, x2, y2)

        
