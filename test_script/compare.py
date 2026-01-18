import numpy as np

file_onnx = "/home/user/Documents/tensorrt_yolo26/build/onnx_output.txt"
file_engine = "/home/user/Documents/tensorrt_yolo26/build/output.txt"

with open(file_onnx, 'r') as f:
    onnx_output = f.readlines()
with open(file_engine, 'r') as f:
    engine_output = f.readlines()
    
numpy_onnx = np.array([float(x.strip()) for x in onnx_output])
numpy_engine = np.array([float(x.strip()) for x in engine_output])

difference = np.abs(numpy_onnx - numpy_engine)
max_difference = np.max(difference)
mean_difference = np.mean(difference)
min_difference = np.min(difference)

print(f"Max difference between ONNX and Engine outputs: {max_difference}")
print(f"Mean difference between ONNX and Engine outputs: {mean_difference}")
print(f"Min difference between ONNX and Engine outputs: {min_difference}")
