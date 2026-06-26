from typing import Tuple

from PIL import Image


def scale_to_fit(
    img: Image.Image,
    max_size: Tuple[int, int] = (3072, 2048),
    min_size: Tuple[int, int] = (1792, 28),
    grid_size: int = 28,
):
    resample_method = Image.Resampling.LANCZOS

    width, height = img.size

    # Check for empty or invalid image
    if width <= 0 or height <= 0:
        return img

    original_ar = width / height
    current_pixels = width * height
    max_pixels = max_size[0] * max_size[1]
    min_pixels = min_size[0] * min_size[1]

    # 1. Determine ideal float scale based on pixel bounds
    scale = 1.0
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5

    # 2. Convert dimensions to integer "grid blocks"
    # Derive h_blocks from rounded w_blocks to preserve aspect ratio.
    # Independent rounding of both axes can push them in opposite directions
    # (e.g. w rounds up, h rounds down), distorting the ratio.
    w_blocks = max(1, round((width * scale) / grid_size))
    h_blocks = max(1, round(w_blocks * (height / width)))

    # 3. Refinement Loop: Ensure we are under the max limit
    while (w_blocks * h_blocks * grid_size * grid_size) > max_pixels:
        if w_blocks == 1 and h_blocks == 1:
            break

        if w_blocks == 1:
            h_blocks -= 1
            continue
        if h_blocks == 1:
            w_blocks -= 1
            continue

        # Compare distortion: Which move preserves Aspect Ratio better?
        ar_w_loss = abs(((w_blocks - 1) / h_blocks) - original_ar)
        ar_h_loss = abs((w_blocks / (h_blocks - 1)) - original_ar)

        if ar_w_loss < ar_h_loss:
            w_blocks -= 1
        else:
            h_blocks -= 1

    # 4. Calculate final pixel dimensions
    new_width = w_blocks * grid_size
    new_height = h_blocks * grid_size

    # Return original if no changes were needed
    if (new_width, new_height) == (width, height):
        return img

    return img.resize((new_width, new_height), resample=resample_method)


def detect_repeat_token(
    predicted_tokens: str,
    base_max_repeats: int = 4,
    window_size: int = 500,
    cut_from_end: int = 0,
    scaling_factor: float = 3.0,
):
    if cut_from_end > 0:
        predicted_tokens = predicted_tokens[:-cut_from_end]

    for seq_len in range(1, window_size // 2 + 1):
        # Extract the potential repeating sequence from the end
        candidate_seq = predicted_tokens[-seq_len:]

        # Inverse scaling: shorter sequences need more repeats
        max_repeats = int(base_max_repeats * (1 + scaling_factor / seq_len))

        # Count how many times this sequence appears consecutively at the end
        repeat_count = 0
        pos = len(predicted_tokens) - seq_len
        if pos < 0:
            continue

        while pos >= 0:
            if predicted_tokens[pos : pos + seq_len] == candidate_seq:
                repeat_count += 1
                pos -= seq_len
            else:
                break

        if repeat_count > max_repeats:
            return True

    return False
