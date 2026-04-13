import os
import onnx
import numpy as np
import onnxruntime
from utils import preprocess_batch, get_d2s
import cv2


model_path = "/home/user/Documents/tensorrt_yolo26/models/yolo26n_batch5.onnx"
image_folder = "/home/user/Documents/tensorrt_yolo26/images/det"
output_folder = "/home/user/Documents/tensorrt_yolo26/build/det_outputs"

os.makedirs(output_folder, exist_ok=True)
model = onnx.load(model_path)

existing = [o.name for o in model.graph.output]

def print_model_info(model):
    """Print model input/output and intermediate node information"""
    print("\n=== Model Information ===")

    print("\nInputs:")
    for input_tensor in model.graph.input:
        print(f"  - {input_tensor.name}: {input_tensor.type}")

    print("\nOutputs:")
    for output_tensor in model.graph.output:
        print(f"  - {output_tensor.name}: {output_tensor.type}")

    print("\nIntermediate Nodes (first 20):")
    for i, node in enumerate(model.graph.node[:20]):
        print(f"  [{i}] {node.op_type}: {node.name}")
        print(f"      Inputs: {node.input}")
        print(f"      Outputs: {node.output}")

    if len(model.graph.node) > 20:
        print(f"  ... and {len(model.graph.node) - 20} more nodes")

    print(f"\nTotal nodes: {len(model.graph.node)}")



###############################################################################################################

# model input shape and name
input_name = model.graph.input[0].name
input_shape = model.graph.input[0].type.tensor_type.shape
print(f"Input name: {input_name}")
print(f"Input shape: {[dim.dim_value for dim in input_shape.dim]}")

# model output shape and name
for output in model.graph.output:
    output_name = output.name
    output_shape = output.type.tensor_type.shape
    print(f"Output name: {output_name}")
    print(f"Output shape: {[dim.dim_value for dim in output_shape.dim]}")


all_image_paths = [f"{image_folder}/{img_name}" for img_name in os.listdir(image_folder) if img_name.endswith((".jpg", ".png"))]
all_image_paths.sort()

image_data = preprocess_batch(all_image_paths, dst_width=640, dst_height=640, border_pixel_color_value=128)
print(f"Input Image Batch Shape: {image_data.shape}")
print(f"Input Image Batch Dtype: {image_data.dtype}")
print(f"Input Image Batch Min/Max: {image_data.min()}/{image_data.max()}")

sess = onnxruntime.InferenceSession(model_path, None)
input_name = sess.get_inputs()[0].name
output = sess.run([model.graph.output[0].name], {input_name: image_data})[0]

print(f"Model output type: {type(output)}")
print(f"Model output shape: {output.shape}")
print(f"Model output dtype: {output.dtype}")

for i, result in enumerate(output):
    image = cv2.imread(all_image_paths[i])
    height, width = image.shape[:2]
    d2s = get_d2s(width, height, dst_w=640, dst_h=640)
    scale = d2s[0, 0]
    tx = d2s[0, 2]
    ty = d2s[1, 2]
    for det in result:
        bbox = det[:4]
        conf = det[4]
        class_id = det[5].astype(int)
        if conf > 0.5:  # confidence threshold
            x1 = int((bbox[0] - tx) / scale)
            y1 = int((bbox[1] - ty) / scale)
            x2 = int((bbox[2] - tx) / scale)
            y2 = int((bbox[3] - ty) / scale)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"ID:{class_id} Conf:{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(f"{output_folder}/result_{i}.jpg", image)



