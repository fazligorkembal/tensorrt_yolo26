from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="onnx", dynamic=False, simplify=True)
