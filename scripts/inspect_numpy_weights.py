import numpy as np

layer_path = "/home/user/Documents/tensorrt_yolo26/models/_model_23_Constant_12_output_0"
weights = np.load(layer_path)

print(f"Weights shape: {weights.shape}")
print(f"weights[:, 0, :300] =\n{weights[:, 0, :300]} ...")
print(f"min(weights[:, 0, :]) = {np.min(weights[:, 0, :])}")
print(f"mean(weights[:, 0, :]) = {np.mean(weights[:, 0, :])}")
print(f"max(weights[:, 0, :]) = {np.max(weights[:, 0, :])}")

print(f"weights[:, 1, :300] =\n{weights[:, 1, :300]} ...")
print(f"min(weights[:, 1, :]) = {np.min(weights[:, 1, :])}")
print(f"mean(weights[:, 1, :]) = {np.mean(weights[:, 1, :])}")
print(f"max(weights[:, 1, :]) = {np.max(weights[:, 1, :])}")

weights_flatten = weights.flatten()
with open(
    "/home/user/Documents/tensorrt_yolo26/build/onnx_constant_grid.txt", "w"
) as f:
    for value in weights_flatten:
        f.write(f"{value}\n")


layer_path = "/home/user/Documents/tensorrt_yolo26/models/_model_23_Constant_13_output_0"
weights = np.load(layer_path)
print(f"Weights shape: {weights.shape}")
starts = np.where(weights[0, :-1] != weights[0, 1:])[0] + 1
for i in starts:
    print(f"{weights[:, i-1]} -> {weights[:, i]} at index {i}")

weights_flatten = weights.flatten()
with open(
    "/home/user/Documents/tensorrt_yolo26/build/onnx_constant_stride.txt", "w"
) as f:
    for value in weights_flatten:
        f.write(f"{value}\n")
print("Done.")