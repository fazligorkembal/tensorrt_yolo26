from ultralytics import YOLO

model = YOLO("yolo26x.pt")
model.export(format="onnx")
