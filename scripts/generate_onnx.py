from ultralytics import YOLO

model = YOLO("yolo26n-obb.pt")
model.export(format="onnx")
