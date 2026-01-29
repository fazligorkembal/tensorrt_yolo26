import onnx
import numpy as np
import cv2
import onnxruntime

model_path = "/home/user/Documents/tensorrt_yolo26/models/yolo26n-obb.onnx"
new_model_path = (
    "/home/user/Documents/tensorrt_yolo26/build/yolo26n_cropped.onnx"
)
image_path = "/home/user/Documents/tensorrt_yolo26/images/P0005.jpg"
last_layer = "/model.10/cv2/act/Mul_output_0"
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


def list_all_nodes(model):
    """List all node outputs in the model"""
    print("\n=== All Node Outputs ===")
    for i, node in enumerate(model.graph.node):
        print(f"[{i:4d}] {node.op_type:15s} -> {node.output}")


print_model_info(model)
list_all_nodes(model)


if last_layer not in existing:
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(
            last_layer, onnx.TensorProto.FLOAT, None  # shape önemli değil
        )
    )

onnx.save(model, new_model_path)

model = onnx.load(new_model_path)


def preprocess_image(
    image_path,
    d2s,  # 2x3 affine matrix (numpy array)
    dst_width=1024,
    dst_height=1024,
    border_value=128,
):
    # BGR uint8
    img = cv2.imread(image_path)
    src_h, src_w = img.shape[:2]

    # warpAffine
    warped = cv2.warpAffine(
        img,
        d2s,
        (dst_width, dst_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_value, border_value, border_value),
    )

    # BGR -> RGB
    warped = warped[:, :, ::-1]

    # normalize
    warped = warped.astype(np.float32) / 255.0

    # HWC -> CHW
    warped = np.transpose(warped, (2, 0, 1))

    # add batch
    warped = np.expand_dims(warped, axis=0)

    return warped


def get_d2s(src_w, src_h, dst_w=1024, dst_h=1024):
    scale = min(dst_w / src_w, dst_h / src_h)

    tx = -scale * src_w * 0.5 + dst_w * 0.5
    ty = -scale * src_h * 0.5 + dst_h * 0.5

    d2s = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)

    return d2s


for node in model.graph.node:
    print(f"{node.name} -> outputs: {node.output}")

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

# preprocess the image
image = cv2.imread(image_path)
src_h, src_w = image.shape[:2]
d2s = get_d2s(src_w, src_h, 1024, 1024)
image_data = preprocess_image(image_path, d2s)

image_data_flatten = image_data.flatten()
with open(
    "/home/user/Documents/tensorrt_yolo26/build/onnx_input.txt", "w"
) as f:
    for value in image_data_flatten:
        f.write(f"{value}\n")

sess = onnxruntime.InferenceSession(new_model_path, None)
input_name = sess.get_inputs()[0].name
output = sess.run([last_layer], {input_name: image_data})[0]

output_flatten = output.flatten()
np.savetxt(
    "/home/user/Documents/tensorrt_yolo26/build/onnx_output.txt",
    output_flatten,
    fmt="%f",
)

print(
    "Preprocessed image data saved to /home/user/Documents/tensorrt_yolo26/build/onnx_input.txt"
)

print(
    "Cropped ONNX model output saved to /home/user/Documents/tensorrt_yolo26/build/onnx_output.txt"
)
