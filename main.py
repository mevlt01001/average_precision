import json
from image import Frames
#keys: ['ID', 'truth_boxes', 'pred_boxes', 'preprocess_ms', 'inference_ms', 'postprocess_ms', 'gpu_watt_usage', 'gpu_memory_usage', 'ram_memory_usage']

truth_boxes = []
pred_boxes = []
preprocess_ms = []
inference_ms = []
postprocess_ms = [] 
gpu_watt_usage = []
gpu_memory_usage = [] 
ram_memory_usage = []

path = 'rtdetr-l.engine_results.odgt'

data = list[dict]
with open(path, 'r') as f:
    data = [json.loads(line) for line in f]
    
    
for line in data:
    truth_boxes.append(line['truth_boxes']) # [x1, y1, x2, y2, class_id]
    
    pred_data = []
    
    for box in line['pred_boxes']:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        class_id = box[5]
        conf = box[4]
        pred_data.append([x1, y1, x2, y2, class_id, conf])
        
    pred_boxes.append(pred_data)
    preprocess_ms.append(line['preprocess_ms'])
    inference_ms.append(line['inference_ms'])
    postprocess_ms.append(line['postprocess_ms'])
    gpu_watt_usage.append(line['gpu_watt_usage'])
    gpu_memory_usage.append(line['gpu_memory_usage'])
    ram_memory_usage.append(line['ram_memory_usage'])
 
print(f"{path.upper():#^70}")    
frames = Frames(whole_pred_boxes=pred_boxes, whole_truth_boxes=truth_boxes, iou_treshold=0.5)

average_precision = frames.calc_mAP()
average_preprocess_ms = sum(preprocess_ms) / len(preprocess_ms)
average_inference_ms = sum(inference_ms) / len(inference_ms)
average_postprocess_ms = sum(postprocess_ms) / len(postprocess_ms)
fps = 1000 / (average_inference_ms + average_postprocess_ms + average_preprocess_ms)
average_gpu_watt_usage = sum(gpu_watt_usage) / len(gpu_watt_usage)
average_gpu_memory_usage = sum(gpu_memory_usage) / len(gpu_memory_usage)
average_ram_memory_usage = sum(ram_memory_usage) / len(ram_memory_usage)

print(f"mAP: {average_precision}")
print(f"FPS: {fps}")
print(f"average_preprocess_ms: {average_preprocess_ms}")
print(f"average_inference_ms: {average_inference_ms}")
print(f"average_postprocess_ms: {average_postprocess_ms}")
print(f"average_gpu_watt_usage: {average_gpu_watt_usage}")
print(f"average_gpu_memory_usage: {average_gpu_memory_usage}")
print(f"average_ram_memory_usage: {average_ram_memory_usage}")

frames.plot_ap(title="TRT-FP16 Optimized Model Precision-Recall Curve")
