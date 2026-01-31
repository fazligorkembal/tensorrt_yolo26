from ultralytics import YOLO
import cv2

image_folder = "/home/user/Documents/tensorrt_yolo26/images"

def get_all_image_paths(folder_path):
    import os
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths

model = YOLO("yolo26n-obb.pt")

image_paths = get_all_image_paths(image_folder)

for img_path in image_paths:
    img = cv2.imread(img_path)
    results = model.predict(source=img, conf=0.1, save=False)
    annotated_frame = results[0].plot()
    cv2.imwrite("annotated_" + img_path.split("/")[-1], annotated_frame)
    print(f"Processed and saved: annotated_{img_path.split('/')[-1]}")