"""
OpenCV Handwritten Character Segmentation Engine.

Features:
- Grayscale conversion and automatic background contrast thresholding.
- Noise reduction and speckle filtering.
- Connected component & contour detection.
- Intelligent multi-stroke grouping: merges vertical multi-stroke components
  (such as dots of 'i'/'j', disconnected crossbars, accents) based on horizontal overlap
  without merging horizontally adjacent distinct characters.
- Connected character splitting using vertical projection profile analysis for touching glyphs.
- Left-to-right spatial sorting.
- Aspect-ratio preserving 20x20 scaling and 28x28 centering with EMNIST normalization.
- Annotated visualization image with labeled bounding boxes and indices.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

EMNIST_MEAN = 0.1751
EMNIST_STD = 0.3332

TENSOR_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((EMNIST_MEAN,), (EMNIST_STD,))
])


def split_touching_characters(char_binary, min_val_depth=0.7):
    """
    Checks if a wide component is two touching characters and splits at the projection valley.
    Returns list of split binary sub-images, or [char_binary] if not touching.
    """
    h, w = char_binary.shape
    # Standard alphanumeric aspect ratio in isolated bounding boxes is w/h <= 1.25
    if w <= 1.35 * h or w < 30:
        return [char_binary]

    # Calculate vertical projection profile (number of foreground pixels in each column)
    proj = np.sum(char_binary > 0, axis=0).astype(float)
    if len(proj) == 0:
        return [char_binary]

    # Search for valley in the central region (28% to 72% of width)
    start_col = int(0.28 * w)
    end_col = int(0.72 * w)
    if start_col >= end_col:
        return [char_binary]

    central_proj = proj[start_col:end_col]
    min_idx = np.argmin(central_proj)
    split_col = start_col + min_idx
    min_val = central_proj[min_idx]
    mean_val = np.mean(proj)

    # If the valley is significantly thinner than surrounding columns, split it
    if min_val <= min_val_depth * mean_val or min_val <= 3:
        left = char_binary[:, :split_col]
        right = char_binary[:, split_col:]
        # Ensure both parts have non-trivial ink
        if cv2.countNonZero(left) >= 15 and cv2.countNonZero(right) >= 15:
            return [left, right]

    return [char_binary]


def segment_and_preprocess_canvas(pil_img, min_area=25):
    """
    Segments handwritten text from a canvas into individual characters.

    Args:
        pil_img (PIL.Image): Drawing canvas image.
        min_area (int): Minimum pixel area to ignore dust or tiny accidental clicks.

    Returns:
        segments (list of dict): Each dict contains:
            - 'tensor': [1, 1, 28, 28] PyTorch tensor
            - 'preview_img': 28x28 PIL.Image
            - 'bbox': (x, y, w, h) in original canvas coordinates
            - 'order': int (1-based index from left to right)
        annotated_img (PIL.Image): Visual preview with numbered bounding boxes.
        error_msg (str or None): Error message if no characters detected.
    """
    if pil_img is None:
        return [], None, "No image provided."

    # Convert PIL Image to OpenCV grayscale numpy array
    rgb = np.array(pil_img.convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Invert so ink is white (255) on black background (0)
    corners = [float(gray[0, 0]), float(gray[0, -1]), float(gray[-1, 0]), float(gray[-1, -1])]
    if np.mean(corners) > 128.0:
        binary = cv2.bitwise_not(gray)
    else:
        binary = gray.copy()

    # Threshold small noise
    _, thresh = cv2.threshold(binary, 30, 255, cv2.THRESH_BINARY)

    # Check total ink mass
    if cv2.countNonZero(thresh) < min_area:
        return [], None, "Canvas is empty. Draw one or more characters before recognizing."

    # Find connected components with statistics
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    raw_boxes = []
    # Index 0 is background; check labels 1..num_labels-1
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area and (w >= 3 or h >= 3):
            raw_boxes.append([x, y, w, h])

    if not raw_boxes:
        return [], None, "No distinct characters detected. Please write clearly."

    # Sort bounding boxes initially from left to right
    raw_boxes.sort(key=lambda b: b[0])

    # -------------------------------------------------------------
    # Step 1: Multi-Stroke Grouping (Vertical / Overlapping Strokes)
    # Merge strokes only if they overlap significantly in X
    # (e.g. dot of 'i'/'j', vertical bars, stacked strokes)
    # -------------------------------------------------------------
    merged_boxes = []
    while raw_boxes:
        curr = raw_boxes.pop(0)
        cx, cy, cw, ch = curr
        merged_any = True

        while merged_any:
            merged_any = False
            i = 0
            while i < len(raw_boxes):
                nx, ny, nw, nh = raw_boxes[i]

                # Calculate horizontal overlap
                x_overlap = max(0, min(cx + cw, nx + nw) - max(cx, nx))
                min_w = min(cw, nw)

                # Check if one component's horizontal center is within the other's span
                c_center_x = cx + cw / 2.0
                n_center_x = nx + nw / 2.0
                center_inside = (cx <= n_center_x <= cx + cw) or (nx <= c_center_x <= nx + nw)

                # Merge ONLY if they share significant horizontal overlap
                # (e.g. dot above stem, or two strokes of same letter)
                if x_overlap > 0.3 * min_w or center_inside:
                    # Verify they don't form an unnaturally wide block
                    new_min_x = min(cx, nx)
                    new_max_x = max(cx + cw, nx + nw)
                    new_w = new_max_x - new_min_x

                    new_min_y = min(cy, ny)
                    new_max_y = max(cy + ch, ny + nh)
                    new_h = new_max_y - new_min_y

                    if new_w <= 1.4 * new_h:
                        cx, cy, cw, ch = new_min_x, new_min_y, new_w, new_h
                        raw_boxes.pop(i)
                        merged_any = True
                        continue

                i += 1

        merged_boxes.append((cx, cy, cw, ch))

    # Sort strictly from left to right
    merged_boxes.sort(key=lambda b: b[0])

    # -------------------------------------------------------------
    # Step 2: Check for Touching Characters & Extract BBoxes
    # -------------------------------------------------------------
    final_character_crops = []
    for (bx, by, bw, bh) in merged_boxes:
        pad = 2
        x0 = max(0, bx - pad)
        y0 = max(0, by - pad)
        x1 = min(thresh.shape[1], bx + bw + pad)
        y1 = min(thresh.shape[0], by + bh + pad)

        char_crop = thresh[y0:y1, x0:x1]

        # Check for touching characters
        sub_crops = split_touching_characters(char_crop)
        if len(sub_crops) > 1:
            # Splitted into 2 characters
            offset = 0
            for sc in sub_crops:
                sw = sc.shape[1]
                final_character_crops.append((sc, (x0 + offset, y0, sw, bh)))
                offset += sw
        else:
            final_character_crops.append((char_crop, (bx, by, bw, bh)))

    # Sort final character crops by x position
    final_character_crops.sort(key=lambda item: item[1][0])

    # -------------------------------------------------------------
    # Step 3: Aspect-Ratio Preserving 20x20 Centering on 28x28 Canvas
    # -------------------------------------------------------------
    segments = []
    annotated_rgb = rgb.copy()

    for idx, (crop_mask, (bx, by, bw, bh)) in enumerate(final_character_crops):
        coords = np.argwhere(crop_mask > 30)
        if coords.size < 12:
            continue

        cy0, cx0 = coords.min(axis=0)
        cy1, cx1 = coords.max(axis=0) + 1
        tight_crop = Image.fromarray(crop_mask[cy0:cy1, cx0:cx1])

        # Scale preserving aspect ratio inside 20x20
        tw, th = tight_crop.size
        max_dim = max(tw, th)
        scale = 20.0 / max_dim
        new_w = max(1, int(round(tw * scale)))
        new_h = max(1, int(round(th * scale)))

        resized = tight_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Center on 28x28 black canvas
        canvas_28 = Image.new('L', (28, 28), 0)
        offset_x = (28 - new_w) // 2
        offset_y = (28 - new_h) // 2
        canvas_28.paste(resized, (offset_x, offset_y))

        # PyTorch Tensor with EMNIST standardization
        tensor = TENSOR_TRANSFORM(canvas_28).unsqueeze(0)  # [1, 1, 28, 28]

        segments.append({
            "tensor": tensor,
            "preview_img": canvas_28,
            "bbox": (int(bx), int(by), int(bw), int(bh)),
            "order": idx + 1
        })

        # Draw bounding box and index on annotated preview
        # Primary warm brown box (#897765 -> RGB: 137, 119, 101)
        cv2.rectangle(annotated_rgb, (bx, by), (bx + bw, by + bh), (137, 119, 101), 2)
        cv2.rectangle(annotated_rgb, (bx, max(0, by - 16)), (bx + 16, max(0, by)), (137, 119, 101), -1)
        cv2.putText(annotated_rgb, str(idx + 1), (bx + 3, max(0, by - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    if not segments:
        return [], None, "Failed to segment clean characters. Please write clearly."

    annotated_img = Image.fromarray(annotated_rgb)
    return segments, annotated_img, None
