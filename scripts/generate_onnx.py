from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="onnx", batch=4, name="yolo26n_batch4", project="/home/user/Documents/tensorrt_yolo26/models")
