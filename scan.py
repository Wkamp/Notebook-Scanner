import cv2 as cv
import numpy as np
import argparse

from scipy.ndimage import distance_transform_edt

from autocrop import crop_to_paper
from extract_ink import (
    classify_mask_colors,
    cc_false_positive_cleanup,
    cleanup_intersection_black,
)


def shadow_removal(img):
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    l, a, b = cv.split(lab)

    kernel_size = int(img.shape[1] / 4)
    if kernel_size % 2 == 0:
        kernel_size += 1
    bg_illumination = cv.GaussianBlur(l, (kernel_size, kernel_size), 0)

    # rescaling to 100 makes histogram more bimodal, making thresholding easier
    l_norm_threshold = cv.divide(l, bg_illumination, scale=100)

    # scale=150 looks better when using for black ink
    l_norm_black_output = cv.divide(l, bg_illumination, scale=150)

    return cv.merge([l_norm_threshold.astype(np.uint8), a, b]), cv.merge(
        [l_norm_black_output.astype(np.uint8), a, b]
    )


# creates preview window so user can see threshold mask
def manual_mask(gray_clean, initial_threshold):
    if initial_threshold is None:
        initial_threshold = 70

    window_name = "Threshold Preview: [S]ave & Exit, [Q]uit"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.createTrackbar("Threshold", window_name, initial_threshold, 255, lambda x: None)

    while True:
        t = cv.getTrackbarPos("Threshold", window_name)
        _, preview_mask = cv.threshold(gray_clean, t, 255, cv.THRESH_BINARY_INV)
        cv.imshow(window_name, preview_mask)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            cv.destroyAllWindows()

            return None, None

        if key == ord("s"):
            cv.destroyAllWindows()
            _, final_mask = cv.threshold(gray_clean, t, 255, cv.THRESH_BINARY_INV)
            return final_mask, t

    # this shouldn't be reached, but just in-case
    cv.destroyAllWindows()
    return None, None


def cmd_arguments():
    parser = argparse.ArgumentParser(
        description="Ink extractor with auto or manual thresholding."
    )
    parser.add_argument("input", help="Path to input image")
    parser.add_argument(
        "threshold",
        type=int,
        nargs="?",
        default=None,
        help="Manual luma threshold for ink masking instead of auto detection. If in manual mode this will be used for default luma",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Interactive binary threshold slider instead of auto detection",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Output .png with transparent background instead of white",
    )
    parser.add_argument(
        "--colors",
        type=str,
        default="bl",
        help="Comma-separated ink colors to detect. Options: r, g, b, bl (r=red, g=green, b=blue, bl=black)",
    )
    parser.add_argument(
        "--crop_only",
        action="store_true",
        help="Only crops to notebook, no further processing.",
    )
    parser.add_argument(
        "--ink",
        type=str,
        default="std",
        help="Determines the color of ink mask. \nOptions: og (samples image ink),\ngray (grayscale image ink),\nbw (ink is black),\nstd (default, ink color is between original and grayscale)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ink",
        help="Provide filename output excluding extension (always .png)",
    )

    return parser


def validate_input_colors(parser, args):
    VALID_COLORS = {"r", "g", "b", "bl"}
    active_colors = [c.strip() for c in args.colors.split(",")]

    invalid = [c for c in active_colors if c not in VALID_COLORS]
    if invalid:
        parser.error(f"Invalid color(s): {invalid}. Choose from: r, g, b, bl")

    if len(active_colors) < 1:
        parser.error("At least one color must be specified.")

    return active_colors


def write_output_img(img, filename, transparent):
    # converts white pixels to transparent background
    if transparent:
        if len(img.shape) == 2:
            print("Error: Cannot make grayscale image transparent")
            return

        alpha_channel = np.where(np.all(img == 255, axis=2), 0, 255).astype(np.uint8)
        bgra = cv.merge(
            [
                img[:, :, 0],
                img[:, :, 1],
                img[:, :, 2],
                alpha_channel,
            ]
        )
        cv.imwrite(f"{filename}.png", bgra)

        print(f"Saved Image as: {filename}.png")
        return

    cv.imwrite(f"{filename}.png", img)
    print(f"Saved Image as: {filename}.png")


def main():
    parser = cmd_arguments()
    args = parser.parse_args()

    img_raw = cv.imread(args.input)
    if img_raw is None:
        print("Error: Could not read image.")
        return

    # colors must be in [red, green, blue, black]
    active_colors = validate_input_colors(parser, args)
    print(f"Detecting Colors: {active_colors}\n")

    paper = crop_to_paper(img_raw)
    if args.crop_only:
        write_output_img(paper, args.output, transparent=False)
        return

    # helps with thresholding and anchoring black color class
    print("Removing Image Shadows...")
    lab_img, lab_img_black_output = shadow_removal(paper)
    img_shadow_removed = cv.cvtColor(lab_img, cv.COLOR_Lab2BGR)

    black_pen_sample_img = cv.cvtColor(lab_img_black_output, cv.COLOR_Lab2BGR)
    gray = cv.cvtColor(img_shadow_removed, cv.COLOR_BGR2GRAY)
    gray_clean = cv.medianBlur(gray, 3)

    # provides interactive slider for user to pick luma threshold
    if args.manual:
        print("Please select luma threshold: 's' to select threshold, 'q' to quit.")
        ink_mask, chosen_threshold = manual_mask(gray_clean, args.threshold)

        # happens if user quits out of interactive manual_mask
        if ink_mask is None:
            print("Exiting...")
            return

        print(f"Detecting Ink with Threshold={chosen_threshold}...")

    elif args.threshold is not None:
        print(f"Detecting Ink With Threshold: {args.threshold}...")

        _, ink_mask = cv.threshold(
            gray_clean, args.threshold, 255, cv.THRESH_BINARY_INV
        )
    else:
        print("Detecting Ink...")
        _, ink_mask = cv.threshold(
            gray_clean, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
        )

    # uses kmeans++ over a,b space
    print("Determining Ink Colors...")
    colored_ink_img, mask = classify_mask_colors(
        lab_img, ink_mask, active_colors=active_colors
    )

    # classes won't be perfect for example:
    #   concentrated globs/ink of any color is essentially black
    print("Refining Ink...")
    colored_ink_img = cv.medianBlur(colored_ink_img, 3)
    mask = np.any(colored_ink_img != 255, axis=2)  # updates mask

    # dilates to fill small gaps created from processing/incomplete masking
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    dilated_mask = cv.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    new_pixels = dilated_mask & ~mask

    # maps dilated pixels to the nearest neighbor color
    # if nn mapping is wrong it will be cleaned up by next step
    _, nearest_idx = distance_transform_edt(~mask, return_indices=True)
    colored_ink_img[new_pixels] = colored_ink_img[
        nearest_idx[0][new_pixels], nearest_idx[1][new_pixels]
    ]

    mask = dilated_mask
    mask = np.any(colored_ink_img != 255, axis=2)

    # gets rid of the majority of misclassifications
    cleaned = cc_false_positive_cleanup(
        colored_ink_img, mask, max_area=100, neighbor_radius=8, recolor_threshold=0.6
    )

    # get rid of very small remaining misclassifications
    cleaned = cc_false_positive_cleanup(
        cleaned, mask, max_area=15, neighbor_radius=3, recolor_threshold=0.9
    )

    # this removes the black at color intersections
    # (this isn't really necessary, I just think it looks nicer)
    cleaned = cleanup_intersection_black(cleaned, mask, 200, 2, 10)

    # white background
    background_img = np.full(cleaned.shape, 255, dtype=np.uint8)
    mask_idxs = np.where(mask > 0)

    if len(active_colors) == 1 and active_colors[0] == "bl":
        # default case
        background_img[mask_idxs] = black_pen_sample_img[mask_idxs]

        match args.ink:
            case "og":
                background_img[mask_idxs] = paper[mask_idxs]
            case "gray":
                background_img = cv.cvtColor(background_img, cv.COLOR_BGR2GRAY)
                background_img[mask_idxs] = gray[mask_idxs]
            case "bw":
                background_img = cv.cvtColor(background_img, cv.COLOR_BGR2GRAY)
                background_img[mask_idxs] = 0
            case "std":  # redundant, but to be explicit
                background_img[mask_idxs] = black_pen_sample_img[mask_idxs]

        cleaned = background_img

    write_output_img(cleaned, args.output, args.transparent)


if __name__ == "__main__":
    main()
