import cv2 as cv
import numpy as np

from scipy.optimize import linear_sum_assignment


def classify_mask_colors(img, mask, active_colors=None):
    ALL_colors = {"r": "red", "g": "green", "b": "blue", "bl": "black"}

    # defaults to using all colors if none provided
    if active_colors is None:
        active_colors = list(ALL_colors.keys())

    color_names = [ALL_colors[c] for c in active_colors]

    mask_indices = np.where(mask > 0)
    pixel_values = img[mask_indices].astype(np.float32)

    if len(pixel_values) < len(active_colors):
        return np.zeros_like(img), mask

    # k-means only done in a,b channels
    ab_values = pixel_values[:, 1:]

    # k-means++ clustering
    k = len(active_colors)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers_ab = cv.kmeans(
        ab_values, k, None, criteria, 10, cv.KMEANS_PP_CENTERS
    )

    # range of r,g,b in a,b space (for cluster mapping)
    ALL_REFERENCES = {
        "red": np.array([170.0, 159.0]),
        "green": np.array([114.0, 131.0]),
        "blue": np.array([135.0, 110.0]),
    }

    # clusters are mapped to these labels
    ALL_COLOR_MAP = {
        "black": [0, 0, 0],
        "red": [0, 0, 255],
        "green": [0, 255, 0],
        "blue": [255, 0, 0],
    }

    cluster_L = {}
    for i in range(k):
        cluster_pixels = pixel_values[labels.flatten() == i]
        cluster_L[i] = np.mean(cluster_pixels[:, 0]) if len(cluster_pixels) > 0 else 255

    c_map = {}
    used_color_names = set()

    # label black cluster first
    if "black" in color_names:
        NEUTRAL_AB = np.array([128.0, 128.0])
        black_scores = {}
        for i in range(k):
            ab_neutrality = np.linalg.norm(centers_ab[i] - NEUTRAL_AB)

            # should have low luma and be neutral in a,b space
            black_scores[i] = cluster_L[i] + (0.5 * ab_neutrality)

        black_cluster_idx = min(black_scores, key=black_scores.get)
        c_map[black_cluster_idx] = ALL_COLOR_MAP["black"]
        used_color_names.add("black")

    # use hungarian algorithm to map the remaining clusters
    remaining_cluster_indices = [i for i in range(k) if i not in c_map]
    target_names = [name for name in ["red", "green", "blue"] if name in color_names]

    if remaining_cluster_indices and target_names:
        cost_matrix = []
        for c_idx in remaining_cluster_indices:
            row = []
            for t_name in target_names:
                dist = np.linalg.norm(centers_ab[c_idx] - ALL_REFERENCES[t_name])
                row.append(dist)
            cost_matrix.append(row)

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        for r_idx, c_idx in zip(row_indices, col_indices):
            actual_cluster_idx = remaining_cluster_indices[r_idx]
            assigned_color_name = target_names[c_idx]
            c_map[actual_cluster_idx] = ALL_COLOR_MAP[assigned_color_name]

    # mask color map
    final_colors = np.array(
        [c_map.get(label, [255, 255, 255]) for label in labels.flatten()],
        dtype=np.uint8,
    )

    # set background to white
    colored_img = np.full(img.shape, 255, dtype=np.uint8)

    # map mask to correct colors
    colored_img[mask_indices] = final_colors

    return colored_img, mask


# gets rid of color mislabels, when surrounded by correct color
def cc_false_positive_cleanup(
    result_img,
    ink_mask,
    max_area=100,
    neighbor_radius=16,
    recolor_threshold=0.75,
    colors={
        "black": np.array([0, 0, 0], dtype=np.uint8),
        "red": np.array([0, 0, 255], dtype=np.uint8),
        "green": np.array([0, 255, 0], dtype=np.uint8),
        "blue": np.array([255, 0, 0], dtype=np.uint8),
    },
):
    output = result_img.copy()
    kernel_size = neighbor_radius * 2 + 1

    # precalculate vote maps for all colors
    vote_maps = {}
    for name, bgr_val in colors.items():
        mask = np.all(result_img == bgr_val, axis=2).astype(np.float32)
        vote_maps[name] = cv.boxFilter(
            mask, -1, (kernel_size, kernel_size), normalize=False
        )

    # looks for false positives of all colors
    for target_name, target_bgr in colors.items():
        target_mask = np.all(result_img == target_bgr, axis=2).astype(np.uint8)
        target_mask &= (ink_mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(target_mask)

        # find small ccs of this color
        small_indices = np.where(stats[1:, cv.CC_STAT_AREA] < max_area)[0] + 1

        for cc_id in small_indices:
            cc_mask = labels == cc_id

            # sum votes across all pixels in the component, excluding self-color
            votes = {}
            for color_name, v_map in vote_maps.items():
                if color_name == target_name:
                    continue
                votes[color_name] = np.sum(v_map[cc_mask])

            total_votes = sum(votes.values())
            if total_votes == 0:
                continue

            # replace cc with dominant neighbor color if it meets the threshold
            dominant_neighbor_color = max(votes, key=votes.get)
            if (votes[dominant_neighbor_color] / total_votes) >= recolor_threshold:
                output[labels == cc_id] = colors[dominant_neighbor_color]

    return output


# this reduces/removes black ink caused by colors intersecting
def cleanup_intersection_black(
    result_img, mask, max_area=50, min_colors=2, neighbor_radius=3
):
    target_colors = {"red": [0, 0, 255], "green": [0, 255, 0], "blue": [255, 0, 0]}

    h, w = result_img.shape[:2]
    ink_mask = mask > 0
    black_mask = np.all(result_img == [0, 0, 0], axis=2) & ink_mask

    if not np.any(black_mask):
        return result_img

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (neighbor_radius * 2 + 1,) * 2)
    color_presence_masks = []

    dist_stack = np.full((len(target_colors), h, w), 1e6, dtype=np.float32)
    color_vals = np.array(list(target_colors.values()), dtype=np.uint8)

    for i, (name, val) in enumerate(target_colors.items()):
        c_mask = np.all(result_img == val, axis=2).astype(np.uint8)
        if not np.any(c_mask):
            continue

        # nearest color
        dist_stack[i] = cv.distanceTransform(1 - c_mask, cv.DIST_L2, 3)

        # dilation to see if touches black
        color_presence_masks.append(cv.dilate(c_mask, kernel) > 0)

    if len(color_presence_masks) < min_colors:
        return result_img

    # finds pixels near colors
    intersection_potential = np.sum(color_presence_masks, axis=0) >= min_colors
    valid_black_targets = black_mask & intersection_potential

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        valid_black_targets.astype(np.uint8)
    )

    # filter labels by area
    areas = stats[:, cv.CC_STAT_AREA]
    keep_labels = np.where((areas <= max_area) & (areas > 0))[0]

    # mask of pixels to replace
    final_replace_mask = np.isin(labels, keep_labels)

    # maps intersection black to nearest non-black color
    nearest_idx = np.argmin(dist_stack, axis=0)
    nearest_color_img = color_vals[nearest_idx]

    output = result_img.copy()
    output[final_replace_mask] = nearest_color_img[final_replace_mask]

    return output
