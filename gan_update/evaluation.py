import os.path

import cv2
import numpy as np
from skimage import measure
from skimage.morphology import label
from math import log2
from skimage.transform import resize as img_resize
from PIL import Image

from utils import read_grdecl, _porous_converter_6d_3d


def preprocess_image(image):
    """
    Convert input image to a labeled facies matrix.
    If the image is color (RGB/BGR), map unique colors to integer labels.
    Returns a 2D numpy array of facies labels.
    """
    # If image is a file path (string), read it
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image.copy()

    if img.ndim == 3:
        # Convert to RGB for consistency
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = img_rgb.reshape(-1, 3)
        # Find unique colors
        unique_colors, inverse_idx = np.unique(pixels, axis=0, return_inverse=True)
        label_img = inverse_idx.reshape(img_rgb.shape[0], img_rgb.shape[1])
        # Now label_img has integers (0 to n_fac-1) for each unique color/facies
    else:
        label_img = img.copy()
    return label_img


def compute_facies_transition_score(label_img):
    """
    Compute a score based on facies transition probabilities.
    Calculates adjacency frequency and entropy of transitions.
    Higher score if transitions are structured (low entropy, high self-continuity).
    """
    n_rows, n_cols = label_img.shape
    facies_ids = np.unique(label_img)
    n_facies = len(facies_ids)
    # Initialize transition count matrix
    transition_counts = np.zeros((n_facies, n_facies), dtype=int)
    # Iterate through interior pixels to count transitions (4-neighborhood: right and down to avoid double-count)
    for i in range(n_rows):
        for j in range(n_cols):
            current = label_img[i, j]
            idx_cur = np.where(facies_ids == current)[0][0]
            # Right neighbor
            if j < n_cols - 1:
                right = label_img[i, j + 1]
                idx_right = np.where(facies_ids == right)[0][0]
                transition_counts[idx_cur, idx_right] += 1
                transition_counts[idx_right, idx_cur] += 1  # count both directions
            # Down neighbor
            if i < n_rows - 1:
                down = label_img[i + 1, j]
                idx_down = np.where(facies_ids == down)[0][0]
                transition_counts[idx_cur, idx_down] += 1
                transition_counts[idx_down, idx_cur] += 1
    # Calculate probabilities from counts
    transition_prob = transition_counts / transition_counts.sum()
    # Compute entropy of the transition probability distribution
    eps = 1e-12  # small value to avoid log(0)
    entropy = -np.sum(transition_prob * np.log2(transition_prob + eps))
    max_entropy = log2(n_facies * n_facies)  # maximum entropy if all transitions equally likely
    entropy_ratio = entropy / max_entropy
    # Also compute self-transition proportion (how often a pixel is adjacent to same facies)
    self_transitions = 0
    total_neighbors = 0
    # Count self transitions (same facies on either side horizontally or vertically)
    for i in range(n_rows):
        for j in range(n_cols):
            if j < n_cols - 1:
                total_neighbors += 1
                if label_img[i, j] == label_img[i, j + 1]:
                    self_transitions += 1
            if i < n_rows - 1:
                total_neighbors += 1
                if label_img[i, j] == label_img[i + 1, j]:
                    self_transitions += 1
    self_ratio = self_transitions / max(1, total_neighbors)
    # Define transition score: we want low entropy (structured) and moderate/high self continuity
    # For example, we can take (1 - entropy_ratio) weighted with self_ratio.
    transition_score = (1 - entropy_ratio) * 0.5 + self_ratio * 0.5
    # Normalize to 0-1 (it should already be in 0-1 range logically)
    transition_score = max(0.0, min(1.0, transition_score))
    return transition_score


def compute_channel_shape_score(label_img, channel_label=None):
    """
    Compute a score evaluating channel body shapes.
    Identifies connected regions of the channel facies and measures their geometry.
    If channel_label is None, it will attempt to infer the channel facies as the one with largest horizontal extent or minority proportion.
    """
    # Determine channel label if not provided
    if channel_label is None:
        # Heuristic: channel facies might be the one with largest aspect ratio region or smaller proportion of total area
        labels, counts = np.unique(label_img, return_counts=True)
        # For low N/G, channel is minority, so pick smallest count (but >0)
        channel_label = labels[np.argmin(counts)]
    # Create a binary mask for channel facies
    channel_mask = (label_img == channel_label).astype(np.uint8)
    # Label connected components on the channel mask
    # Use 8-connectivity to allow diagonal connection if needed
    labeled_channels = label(channel_mask, connectivity=2)
    regions = measure.regionprops(labeled_channels)
    if len(regions) == 0:
        # No channel present, return low score (or 0.5 neutral, depending on desired behavior)
        return 0.0
    # Metrics to collect
    aspect_ratios = []
    solidities = []
    orientations = []
    for region in regions:
        if region.area < 5:  # skip tiny artifacts/noise if any
            continue
        # Compute aspect ratio: major_axis_length/minor_axis_length (if minor_axis_length is 0, skip region)
        if region.minor_axis_length == 0:
            aspect = region.major_axis_length  # very thin line
        else:
            aspect = region.major_axis_length / region.minor_axis_length
        aspect_ratios.append(aspect)
        # Solidity (area / convex area)
        solidities.append(region.solidity)
        # Orientation in degrees (region.orientation is in radians measuring angle from horizontal axis)
        theta = region.orientation  # angle in radians from horizontal
        orientations.append(np.degrees(theta))
    if len(aspect_ratios) == 0:
        return 0.0
    # Calculate average or median metrics
    avg_aspect = np.mean(aspect_ratios)
    avg_solidity = np.mean(solidities) if solidities else 1.0
    # We expect aspect ratios to be relatively high (e.g. ~5-15) and solidity close to 1 for smooth shapes
    # Normalize aspect: if avg_aspect >= 5, consider it good (cap influence at, say, 15 for very high)
    aspect_score = min(avg_aspect / 15.0, 1.0)  # scale such that 15 or above gives ~1.0 score
    # Solidity: if avg_solidity >= 0.9, very smooth shapes
    solidity_score = avg_solidity  # already between 0 and 1
    # We could also check orientation distribution, but for simplicity, skip or ensure not vertical
    # Combine shape metrics (weight aspect and solidity equally for shape quality)
    shape_score = 0.5 * aspect_score + 0.5 * solidity_score
    shape_score = max(0.0, min(1.0, shape_score))
    return shape_score


def compute_layer_continuity_score(label_img):
    """
    Compute a score for lateral layer continuity.
    Assesses how continuous the dominant facies layers are across the image.
    Uses horizontal run length and facies changes per row as proxies.
    """
    n_rows, n_cols = label_img.shape
    # Identify the dominant facies (most widespread facies by area)
    labels, counts = np.unique(label_img, return_counts=True)
    dominant_label = labels[np.argmax(counts)]
    dominant_mask = (label_img == dominant_label).astype(np.uint8)
    # Metrics: track continuous run lengths of the dominant facies in each row
    run_lengths = []
    facies_changes = []  # number of segments per row
    for i in range(n_rows):
        row = label_img[i, :]
        # find runs of the same facies in this row
        prev_val = row[0]
        run_len = 1
        segments = 1  # at least one segment exists
        max_run_len = 0
        for j in range(1, n_cols):
            if row[j] == row[j - 1]:
                run_len += 1
            else:
                # end of a segment
                if row[j - 1] == dominant_label:
                    # record run length if it was the dominant facies
                    max_run_len = max(max_run_len, run_len)
                run_len = 1
                segments += 1
                prev_val = row[j]
        # Check last segment in row
        if row[-1] == dominant_label:
            max_run_len = max(max_run_len, run_len)
        run_lengths.append(max_run_len)
        facies_changes.append(segments - 1)  # changes = segments-1
    # Now compute continuity metrics
    avg_max_run = np.mean(run_lengths)  # average longest continuous span of dominant facies per row
    avg_changes = np.mean(facies_changes)  # average facies changes per row
    # Normalize metrics: longer runs and fewer changes -> higher continuity score
    run_score = avg_max_run / n_cols  # fraction of row on average covered by continuous dominant facies
    change_score = 1 - (avg_changes / (n_cols - 1))  # if on average many changes, score lowers
    change_score = max(0.0, change_score)  # ensure non-negative
    # Composite continuity score (weight runs and changes equally)
    continuity_score = 0.5 * run_score + 0.5 * change_score
    continuity_score = max(0.0, min(1.0, continuity_score))
    return continuity_score


def evaluate_geological_realism(image, channel_label=None):
    """
    Compute the overall geological realism score for a facies image.
    - image: input image (file path or array) of facies.
    - channel_label: optional label value for channel facies if known.
    Returns a realism score between 0 (lowest realism) and 100 (highest realism).
    """
    label_img = preprocess_image(image)
    # Compute sub-scores
    transition_score = compute_facies_transition_score(label_img)
    shape_score = compute_channel_shape_score(label_img, channel_label=channel_label)
    continuity_score = compute_layer_continuity_score(label_img)
    # Combine scores (equal weighting)
    overall_score_normalized = (transition_score + shape_score + continuity_score) / 3.0
    realism_score = overall_score_normalized * 100  # scale to 0-100
    print(realism_score)
    return realism_score


def save_array_as_image(array: np.ndarray, filename: str):
    """
    Saves a 3D NumPy array with values in [0, 1] as an image.

    Parameters:
    - array (np.ndarray): Input array of shape (H, W, 3), with float values in [0, 1].
    - filename (str): Path to save the image, e.g., 'output.png'.

    Raises:
    - ValueError: If the array shape is not suitable for an RGB image.
    """
    # Validate input
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Input array must have shape (H, W, 3) for an RGB image.")

    # Clip values to [0, 1] and scale to [0, 255]
    array_uint8 = (np.clip(array, 0, 1) * 255).astype(np.uint8)

    # Convert to PIL image and save
    img = Image.fromarray(array_uint8)
    img.save(filename)
    print(f"Image saved to {filename}")


def generate_image(filename, do_flip=False):
    porosity, facies = read_grdecl(filename)
    img = _porous_converter_6d_3d(facies, porosity)

    img = np.transpose(img, [2, 0, 1, 3])

    if (do_flip):
        img = np.flip(img, 1)

    image_shape = img.shape

    height = 64
    width = 64
    upscale_factor = 1
    stride_x = 32
    stride_y = 32
    nc = 6

    image_ch, image_size_x, image_size_y, _ = image_shape
    patch_size_x = patch_size_y = height * upscale_factor
    # todo change stride as needed
    # stride_x = stride_y = 1
    n_images_x = (image_size_x - patch_size_x - 1) // stride_x + 1
    n_images_y = (image_size_y - patch_size_y - 1) // stride_y + 1

    total_img_num = n_images_x * n_images_y * 100

    image_folder = r'C:\NORCE_Projects\DISTINGUISH\Temp'
    result_file = os.path.splitext(os.path.basename(filename))[0]
    result_file = os.path.join(image_folder, '{}.csv'.format(result_file))
    result_handler = open(result_file, 'a+')

    local_idx = 0
    for idx_i in range(0, image_size_x - patch_size_x, stride_x):
        for idx_j in range(0, image_size_y - patch_size_y, stride_y):
            for k in range(img.shape[-1]):
                patch = img[:, idx_i:idx_i + patch_size_x, idx_j:idx_j + patch_size_y, k]
                patch = np.transpose(patch[0:3, :, :], axes=(1, 2, 0))
                patch_normalized = np.rot90((np.clip(patch, 0, 1) * 255).astype(np.uint8))
                realism_score = evaluate_geological_realism(patch_normalized)
                result_handler.write('{}\n'.format(realism_score))
                result_handler.flush()

                # patch_normalized = (patch - 0.5) * 2.0
                # patch_normalized[patch_normalized < -1.0] = -1.0
                # patch_normalized[patch_normalized > 1.0] = 1.0
                # image_patch = np.transpose(patch[0:3, :, :], axes=(1, 2, 0))
                # image_patch_normalized = np.transpose(patch_normalized[0:3, :, :], axes=(1, 2, 0))
                #
                # img_name1 = os.path.join(image_folder, '{}-1.png'.format(local_idx))
                # img_name2 = os.path.join(image_folder, '{}-2.png'.format(local_idx))
                #
                # save_array_as_image(image_patch, img_name1)
                # save_array_as_image(image_patch_normalized, img_name2)
                local_idx += 1

    result_handler.close()


def main():
    files = [r'T:\600\60010\FakeImageDataset\Distinguish\NOFAULT_MODEL_R1.grdecl',
    r'T:\600\60010\FakeImageDataset\Distinguish\NOFAULT_MODEL_R3.grdecl',
    r'T:\600\60010\FakeImageDataset\Distinguish\NOFAULT_MODEL_R4.grdecl',
    r'T:\600\60010\FakeImageDataset\Distinguish\NOFAULT_MODEL_R5.grdecl']
    for file in files:
        generate_image(file)


if __name__ == '__main__':
    main()
