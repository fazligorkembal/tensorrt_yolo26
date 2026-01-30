from ultralytics import YOLO

model = YOLO("yolo26l-obb.pt")
model.export(format="onnx")
