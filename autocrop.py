import cv2 as cv
import numpy as np


# find notebook binding rings in region
def detect_ring_band(region):
    _, region_width = region.shape[:2]

    col_sums = region.sum(axis=0).astype(np.float64)
    search_cols = len(col_sums)

    # calculate rolling variance
    window = 10
    rolling_var = np.array(
        [col_sums[max(0, x - window) : x + window].var() for x in range(search_cols)]
    )

    var_mean = rolling_var.mean()
    var_std = rolling_var.std()
    high_var_threshold = var_mean + (2 * var_std)

    # isolates areas with high variance
    high_var_indices = np.where(rolling_var > high_var_threshold)[0]

    if len(high_var_indices) == 0:
        return None

    ring_end = int(high_var_indices[-1])

    # probably false positive if covers most of region
    if ring_end > int(region_width * 0.90):
        return None

    return ring_end


# searches lhs and rhs for notebook binding rings to crop
def crop_rings(warped):
    h, w = warped.shape[:2]

    search_cols = int(w * 0.15)  # search 15% of image from lhs/rhs
    gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

    # search lhs
    lhs_region = gray[:, :search_cols]
    ring_end = detect_ring_band(lhs_region)
    if ring_end is not None:
        return warped[:, ring_end + 6 :]

    # flips twice to align rhs region like lhs region
    rhs_region = gray[:, w - search_cols :][:, ::-1]

    ring_end = detect_ring_band(rhs_region)
    if ring_end is not None:
        return warped[:, : w - search_cols + (search_cols - ring_end) - 6]

    return warped


# helper that orders quadrilateral points
def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# converts to top-down view
def transform_paper_perspective(contour, img, margin):
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)

    # convert to four quadrilateral points
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv.minAreaRect(contour)
        pts = cv.boxPoints(rect).astype(np.float32)

    pts = order_points(pts)  # convert boundary points to consistent order
    tl, tr, br, bl = pts

    # img width
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    output_width = int(max(width_top, width_bot))

    # img height
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    output_height = int(max(height_left, height_right))

    if output_width == 0 or output_height == 0:
        return img

    # destination image
    output = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )

    M = cv.getPerspectiveTransform(pts, output)
    warped = cv.warpPerspective(img, M, (output_width, output_height))

    # small crop around image edges to get rid of small background in border
    if output_width > 2 * margin and output_height > 2 * margin:
        warped = warped[margin:-margin, margin:-margin]

    return warped


# uses paper statistics to conform contour to paper
def refine_paper_contours(paper_contour, gray, img):

    # rough paper shape
    rough_hull = cv.convexHull(paper_contour)
    internal_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv.fillConvexPoly(internal_mask, rough_hull, 255)

    # erode to get into paper
    kernel_erode = cv.getStructuringElement(cv.MORPH_RECT, (21, 21))
    sample_mask = cv.erode(internal_mask, kernel_erode)

    # calculate color of paper
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mean_hsv = cv.mean(hsv, mask=sample_mask)
    paper_s = mean_hsv[1]
    paper_v = mean_hsv[2]

    # threshold to determine if a region is paper
    lower_bound = np.array([0, 0, max(0, paper_v - 50)])
    upper_bound = np.array([180, min(255, paper_s + 40), 255])
    stat_mask = cv.inRange(hsv, lower_bound, upper_bound)

    # only look inside rough boundary for paper (ignore background)
    refined_mask = cv.bitwise_and(internal_mask, stat_mask)

    kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    kernel_final_close = cv.getStructuringElement(cv.MORPH_RECT, (25, 25))

    # clean up mask
    refined_mask = cv.morphologyEx(refined_mask, cv.MORPH_OPEN, kernel_open)
    refined_mask = cv.morphologyEx(refined_mask, cv.MORPH_CLOSE, kernel_final_close)

    refined_contours, _ = cv.findContours(
        refined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    if not refined_contours:
        return None

    largest_refined = max(refined_contours, key=cv.contourArea)

    return largest_refined


# returns cropped paper image, or original image if no crop needed
def crop_to_paper(img):
    img_copy = img.copy()

    # pre-process
    gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)

    # uses morphological gradient instead of canny to find paper

    kernel_edge = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    morph_edges = cv.morphologyEx(blurred, cv.MORPH_GRADIENT, kernel_edge)

    _, edge_bin = cv.threshold(morph_edges, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    closed_edges = cv.morphologyEx(edge_bin, cv.MORPH_CLOSE, kernel_close)

    contours, _ = cv.findContours(
        closed_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return img

    # largest contour
    paper_contour = max(contours, key=cv.contourArea)
    max_area = cv.contourArea(paper_contour)

    img_height, img_width = img.shape[:2]
    total_area = img_height * img_width
    coverage_ratio = max_area / total_area

    if max_area == 0:
        return img

    # no large contours found, presumably b/c paper fill image
    if coverage_ratio < 0.01:
        return img

    # large contour fills most of image, so crop not needed
    if not (0.10 < coverage_ratio < 0.90):
        return img

    refined_contours = refine_paper_contours(paper_contour, gray, img)
    if refined_contours is None:
        return img

    img_boundary = 12  # crop size of masked image edges
    warped = transform_paper_perspective(refined_contours, img, img_boundary)
    if warped is None:
        return img

    return crop_rings(warped)
