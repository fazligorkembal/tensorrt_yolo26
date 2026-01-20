import numpy as np

file_onnx = "/home/user/Documents/tensorrt_yolo26/build/onnx_output.txt"
file_engine = "/home/user/Documents/tensorrt_yolo26/build/output.txt"

with open(file_onnx, "r") as f:
    onnx_output = f.readlines()
with open(file_engine, "r") as f:
    engine_output = f.readlines()

numpy_onnx = np.array([float(x.strip()) for x in onnx_output])
numpy_engine = np.array([float(x.strip()) for x in engine_output])

print(f"Onnx Length: {len(numpy_onnx)}")
print(f"Engine Length: {len(numpy_engine)}")

difference = np.abs(numpy_onnx - numpy_engine)
max_difference = np.max(difference)
mean_difference = np.mean(difference)
min_difference = np.min(difference)

threshold = 1
diff_bigger_than_threshold = difference[difference > threshold]


print(
    f"Number of differences greater than {threshold}: {len(diff_bigger_than_threshold)}"
)
print(
    f"Percentage of differences greater than {threshold}: {len(diff_bigger_than_threshold) / len(numpy_onnx) * 100:.6f}%"
)
print(f"Max difference between ONNX and Engine outputs: {max_difference}")
print(f"Mean difference between ONNX and Engine outputs: {mean_difference}")
print(f"Min difference between ONNX and Engine outputs: {min_difference}")
print(f"ONNX MEAN: {np.mean(numpy_onnx)}")
print(f"ENGINE MEAN: {np.mean(numpy_engine)}")
