import numpy as np
import cv2


def preprocess_image(
    image_path,
    d2s,  # 2x3 affine matrix (numpy array)
    dst_width=1024,
    dst_height=1024,
    border_pixel_color_value=128,
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
        borderValue=(border_pixel_color_value, border_pixel_color_value, border_pixel_color_value),
    )

    # BGR -> RGB
    warped = warped[:, :, ::-1]

    # normalize
    warped = warped.astype(np.float32) / 255.0

    # HWC -> CHW
    warped = np.transpose(warped, (2, 0, 1))

    warped = np.expand_dims(warped, axis=0)

    return warped

def preprocess_batch(image_paths, dst_width=1024, dst_height=1024, border_pixel_color_value=128):
    batch_data = []
    for img_path in image_paths:
        print(f"Processing image: {img_path}")
        image = cv2.imread(img_path)
        src_h, src_w = image.shape[:2]
        d2s = get_d2s(src_w, src_h, dst_width, dst_height)
        img_data = preprocess_image(img_path, d2s, dst_width, dst_height, border_pixel_color_value)
        batch_data.append(img_data)

    batch_data = np.concatenate(batch_data, axis=0)  # (B, C, H, W)
    return batch_data

def get_d2s(src_w, src_h, dst_w=1024, dst_h=1024):
    scale = min(dst_w / src_w, dst_h / src_h)

    tx = -scale * src_w * 0.5 + dst_w * 0.5
    ty = -scale * src_h * 0.5 + dst_h * 0.5

    d2s = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)

    return d2s