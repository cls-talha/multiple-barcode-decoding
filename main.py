import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyzbar import pyzbar

path = "tagss/IMG_20260520_162133_348.jpg.jpeg"
img = cv2.imread(path, 0)

# background illumination and normalize
background = cv2.GaussianBlur(img, (0,0), sigmaX=50)
normalized = cv2.divide(img, background, scale=255)
normalized = cv2.bilateralFilter(normalized, d=5, sigmaColor=30, sigmaSpace=30)

#  enhancement and sharpening
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12,12))
result = clahe.apply(normalized)

# sharp filter
kernel = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
], dtype=np.float32)
result = cv2.filter2D(result, -1, kernel)

#  Vertical Edge Detection
sobelx = cv2.Sobel(result, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobelx = cv2.normalize(sobelx, None, 0, 255, cv2.NORM_MINMAX)

_, binary = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# expanding all bars to join
close_kernel = np.ones((15,15), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 15))
binary = cv2.dilate(binary, merge_kernel, iterations=1)

open_kernel = np.ones((7,7), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

# detection of boxes
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_area = cleaned.shape[0] * cleaned.shape[1]
raw_rects = []
pad_ratio = 0.15

for c in contours:
    area = cv2.contourArea(c)
    if area < img_area * 0.001:
        continue
    
    rect = cv2.minAreaRect(c)
    (cx, cy), (w, h), angle = rect
    if w == 0 or h == 0:
        continue
        
    raw_rects.append(rect)

# nms, many duplicate
def get_rotated_iou(r1, r2):
    pts1 = cv2.boxPoints(r1).astype(np.float32)
    pts2 = cv2.boxPoints(r2).astype(np.float32)
    ret, inter = cv2.intersectConvexConvex(pts1, pts2)
    if ret == 0 or inter is None:
        return 0.0
    inter_area = cv2.contourArea(inter)
    area1 = r1[1][0] * r1[1][1]
    area2 = r2[1][0] * r2[1][1]
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

raw_rects = sorted(raw_rects, key=lambda r: r[1][0] * r[1][1], reverse=True)
keep_rects = []
suppressed = [False] * len(raw_rects)

for i in range(len(raw_rects)):
    if suppressed[i]:
        continue
    keep_rects.append(raw_rects[i])
    for j in range(i + 1, len(raw_rects)):
        if not suppressed[j] and get_rotated_iou(raw_rects[i], raw_rects[j]) > 0.3:
            suppressed[j] = True

# adding tolerance/padding for better crop of bar codes
barcode_boxes = []
for (cx, cy), (w, h), angle in keep_rects:
    padded_rect = ((cx, cy), (w * (1 + pad_ratio), h * (1 + pad_ratio)), angle)
    box = cv2.boxPoints(padded_rect)
    barcode_boxes.append(box)

# wrap perspective ordering points vvimp
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

ordered_boxes = [order_points(box) for box in barcode_boxes]

# correcting perspective to make barcode straight for detection
def extract_barcode(img, ordered_pts):
    (tl, tr, br, bl) = ordered_pts
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    
    if maxWidth < 10 or maxHeight < 10:
        return None
        
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(ordered_pts.astype("float32"), dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
    # Rotate to keep horizontal if vertical
    if maxHeight > maxWidth:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        
    return warped

def decode_crop(crop):
    if crop is None:
        return None
    decoded = pyzbar.decode(crop)
    if decoded:
        return decoded[0].data.decode("utf-8", errors="replace")
    
    resized = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    decoded = pyzbar.decode(resized)
    if decoded:
        return decoded[0].data.decode("utf-8", errors="replace")
    return None

#  decoding
extracted_barcodes = []
for pts in ordered_boxes:
    warped = extract_barcode(img, pts)
    text = decode_crop(warped)
    extracted_barcodes.append((pts, warped, text))

vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for pts, warped, text in extracted_barcodes:
    if text is None:
        continue

    pts_int = pts.astype(int)
    cv2.polylines(vis, [pts_int], True, (0, 255, 0), 3)

    tl = pts_int[0]
    text_pos = (max(0, tl[0]-150), max(20, tl[1] - 15))

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(vis, (text_pos[0]-5, text_pos[1]-th-5), (text_pos[0]+tw+5, text_pos[1]+baseline+5), (0, 255, 0), -1)

    cv2.putText(vis, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)


plt.figure(figsize=(12, 16))
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
