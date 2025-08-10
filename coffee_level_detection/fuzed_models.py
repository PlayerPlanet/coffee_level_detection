"""
models.py
Implementations of three coffee fill estimation strategies:
 1) Mask-based segmentation (Mask2Former / transformers pipeline)
 2) Classical CV: contour/color-based segmentation (OpenCV)
 3) Monocular depth-based estimation (MiDaS)
 4) Single-point ToF/LiDAR distance reading model

Each strategy is wrapped in a small class and provides a unified API:
  - .estimate_fill(image_path, **kwargs) -> fill_ratio (0..1) or None

Also includes a small fusion helper to combine estimates.

Examples (at bottom): three short example usages.
"""

from transformers import pipeline, DPTForDepthEstimation, DPTImageProcessor, ZoeDepthForDepthEstimation, ZoeDepthImageProcessor
import numpy as np
import cv2
import torch
import os

# --- 1) Mask-based segmentation model (Mask2Former via transformers) ---
# Name: MaskSegmenterModel

checkpoint = "facebook/mask2former-swin-small-coco-instance"
segmenter = pipeline("image-segmentation", model=checkpoint, use_fast=True)

class MaskSegmenterModel:
    """Estimate coffee fill using an instance segmentation model.
    Uses the provided transformers `segmenter` pipeline to get masks for the pot and coffee.
    Returns fill ratio in [0,1] or None if masks not found.
    """
    def __init__(self, segmenter_pipeline=segmenter, pot_labels=None, coffee_labels=None):
        self.segmenter = segmenter_pipeline
        # Allow label fallbacks; dataset label names vary, try common tokens
        self.pot_labels = pot_labels or ["pot", "coffee pot", "thermos", "kettle"]
        self.coffee_labels = coffee_labels or ["coffee", "espresso", "liquid"]

    def estimate_fill(self, image_path):
        segments = self.segmenter(image_path)
        mask_pot = None
        mask_coffee = None
        for seg in segments:
            label = seg.get("label", "").lower()
            mask = np.array(seg["mask"])  # boolean mask
            if any(tok in label for tok in self.pot_labels) and mask_pot is None:
                mask_pot = mask
            if any(tok in label for tok in self.coffee_labels) and mask_coffee is None:
                mask_coffee = mask
        if mask_pot is None or mask_coffee is None:
            return None
        fill_ratio = float(np.sum(mask_coffee) / np.sum(mask_pot))
        return fill_ratio


# --- 2) Classical CV: Contour and color-based estimation ---
# Name: ContourColorModel

class ContourColorModel:
    """Estimate coffee fill using OpenCV color thresholding and contour geometry.
    Works best when the pot is transparent and coffee is visually darker than background.

    Strategy:
      - Detect pot contour (largest suitable contour with roughly circular/rectangular shape)
      - Within pot ROI, threshold dark brown/black colors (in HSV or LAB)
      - Estimate vertical position of the top of the coffee by scanning rows or using contour bounding box
    """
    def __init__(self, pot_min_area=2000):
        self.pot_min_area = pot_min_area

    def _find_pot_contour(self, img_gray):
        # Use adaptive threshold + morphological ops to highlight pot silhouette
        blur = cv2.GaussianBlur(img_gray, (5,5), 0)
        th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,11,2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # choose largest reasonably-sized contour
        contours = [c for c in contours if cv2.contourArea(c) > self.pot_min_area]
        if not contours:
            return None
        pot = max(contours, key=cv2.contourArea)
        return pot

    def estimate_fill(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        h, w = img.shape[:2]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pot_contour = self._find_pot_contour(img_gray)
        if pot_contour is None:
            return None
        # mask for pot
        mask = np.zeros((h,w), dtype=np.uint8)
        cv2.drawContours(mask, [pot_contour], -1, 255, -1)
        # crop to bounding rect for efficiency
        x,y,ww,hh = cv2.boundingRect(pot_contour)
        roi = img[y:y+hh, x:x+ww]
        roi_mask = mask[y:y+hh, x:x+ww]
        if roi.size == 0:
            return None
        # Convert to HSV and threshold dark brown/black coffee
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # heuristics: coffee is dark with low V, medium-low saturation and brownish hue
        # We'll use two ranges to be robust: dark low-V and brownish
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([179, 255, 70])
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
        # brown range (h around 10-30, sat moderate)
        lower_brown = np.array([5, 50, 30])
        upper_brown = np.array([30, 255, 140])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        coffee_mask = cv2.bitwise_or(mask_dark, mask_brown)
        # only keep pixels that are inside detected pot
        coffee_mask = cv2.bitwise_and(coffee_mask, roi_mask)
        # compute vertical fill: for each column, find highest coffee pixel; then take median
        ys, xs = np.where(coffee_mask>0)
        if len(ys) == 0:
            return 0.0  # no coffee pixels found
        # coordinate system: y increases downwards; top row y=0
        top_of_coffee = ys.min()
        bottom_of_pot = np.where(roi_mask.sum(axis=1)>0)[0].max()
        top_of_pot = np.where(roi_mask.sum(axis=1)>0)[0].min()
        # fill ratio (height of liquid column / interior pot height)
        liquid_height = bottom_of_pot - top_of_coffee
        pot_interior_height = bottom_of_pot - top_of_pot
        if pot_interior_height <= 0:
            return None
        fill_ratio = float(np.clip(liquid_height / pot_interior_height, 0.0, 1.0))
        return fill_ratio


# --- 3) Monocular depth estimation with MiDaS ---
# Name: DepthMiDaSModel

# We'll load MiDaS using torch.hub to keep dependency management simple at runtime.
# The model run returns relative depth; calibration is required to convert to physical height.

class DepthMiDaSModel:
    """Estimate coffee fill using a monocular depth estimator (DPT-Hybrid-MiDaS via transformers)."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def _load_model(self):
        if self.model is not None and self.processor is not None:
            return
        try:
            self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(self.device)
            self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        except Exception as e:
            raise RuntimeError(f"Failed to load DPT-Hybrid-MiDaS model: {e}")

    def estimate_fill(self, image_path, pot_mask=None, calib=None):
        self._load_model()
        import PIL.Image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(img_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
        depth_map = prediction

        if pot_mask is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pot_mask = np.zeros_like(gray, dtype=np.uint8)
            c = ContourColorModel()
            pot_contour = c._find_pot_contour(gray)
            if pot_contour is None:
                pot_mask[:] = 1
            else:
                cv2.drawContours(pot_mask, [pot_contour], -1, 1, -1)
        pot_depths = depth_map[pot_mask.astype(bool)]
        if pot_depths.size == 0:
            return None
        h, w = depth_map.shape
        ys, xs = np.where(pot_mask > 0)
        top_of_pot = ys.min()
        bottom_of_pot = ys.max()
        row_medians = []
        for row in range(top_of_pot, bottom_of_pot + 1):
            vals = depth_map[row, :][pot_mask[row, :] > 0]
            row_medians.append(np.median(vals) if vals.size > 0 else np.nan)
        row_medians = np.array(row_medians)
        grads = np.abs(np.nan_to_num(np.gradient(row_medians)))
        if np.nanmax(grads) == 0:
            surface_row = bottom_of_pot
        else:
            surface_idx = np.nanargmax(grads)
            surface_row = top_of_pot + int(surface_idx)
        liquid_height_px = bottom_of_pot - surface_row
        pot_height_px = bottom_of_pot - top_of_pot
        if pot_height_px <= 0:
            return None
        fill_ratio = float(np.clip(liquid_height_px / pot_height_px, 0.0, 1.0))
        if calib and all(k in calib for k in ("empty_depth", "full_depth", "pot_height_mm")):
            surface_depth_val = np.nanmedian(depth_map[surface_row-1:surface_row+2, :][pot_mask[surface_row-1:surface_row+2, :] > 0])
            t = (surface_depth_val - calib["full_depth"]) / (calib["empty_depth"] - calib["full_depth"] + 1e-9)
            t = np.clip(t, 0.0, 1.0)
            height_mm = t * calib["pot_height_mm"]
            return fill_ratio, height_mm
        return fill_ratio

class ZoeDepthModel:
    """Estimate coffee fill using ZoeDepth monocular depth estimator (Intel/zoedepth-nyu-kitti)."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def _load_model(self):
        if self.model is not None and self.processor is not None:
            return
        try:
            self.model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to(self.device)
            self.processor = ZoeDepthImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        except Exception as e:
            raise RuntimeError(f"Failed to load ZoeDepth model: {e}")

    def estimate_fill(self, image_path, pot_mask=None, calib=None):
        self._load_model()
        import PIL.Image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(img_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
        depth_map = prediction

        if pot_mask is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pot_mask = np.zeros_like(gray, dtype=np.uint8)
            c = ContourColorModel()
            pot_contour = c._find_pot_contour(gray)
            if pot_contour is None:
                pot_mask[:] = 1
            else:
                cv2.drawContours(pot_mask, [pot_contour], -1, 1, -1)
        pot_depths = depth_map[pot_mask.astype(bool)]
        if pot_depths.size == 0:
            return None
        h, w = depth_map.shape
        ys, xs = np.where(pot_mask > 0)
        top_of_pot = ys.min()
        bottom_of_pot = ys.max()
        row_medians = []
        for row in range(top_of_pot, bottom_of_pot + 1):
            vals = depth_map[row, :][pot_mask[row, :] > 0]
            row_medians.append(np.median(vals) if vals.size > 0 else np.nan)
        row_medians = np.array(row_medians)
        grads = np.abs(np.nan_to_num(np.gradient(row_medians)))
        if np.nanmax(grads) == 0:
            surface_row = bottom_of_pot
        else:
            surface_idx = np.nanargmax(grads)
            surface_row = top_of_pot + int(surface_idx)
        liquid_height_px = bottom_of_pot - surface_row
        pot_height_px = bottom_of_pot - top_of_pot
        if pot_height_px <= 0:
            return None
        fill_ratio = float(np.clip(liquid_height_px / pot_height_px, 0.0, 1.0))
        if calib and all(k in calib for k in ("empty_depth", "full_depth", "pot_height_mm")):
            surface_depth_val = np.nanmedian(depth_map[surface_row-1:surface_row+2, :][pot_mask[surface_row-1:surface_row+2, :] > 0])
            t = (surface_depth_val - calib["full_depth"]) / (calib["empty_depth"] - calib["full_depth"] + 1e-9)
            t = np.clip(t, 0.0, 1.0)
            height_mm = t * calib["pot_height_mm"]
            return fill_ratio, height_mm
        return fill_ratio
# --- 4) Simple ToF/LiDAR model ---
# Name: ToFModel

class ToFModel:
    """Estimate coffee fill from single-point distance measurement.

    Assumes a vertically-mounted sensor providing distance to liquid surface (in mm).
    Must be calibrated with known pot height or known empty/full distances.
    """
    def __init__(self, pot_height_mm=None, empty_distance_mm=None, full_distance_mm=None):
        # If pot_height_mm provided, we assume the sensor measures from a fixed mount above pot rim.
        self.pot_height_mm = pot_height_mm
        self.empty_distance_mm = empty_distance_mm
        self.full_distance_mm = full_distance_mm

    def estimate_fill(self, distance_mm=None, distance_measurements=None):
        """
        Provide either a single distance_mm or a list/array of distance_measurements to average.
        Returns fill_ratio in [0,1] and estimated height_mm if pot_height_mm known.
        """
        if distance_mm is None and distance_measurements is not None:
            distance_mm = float(np.median(distance_measurements))
        if distance_mm is None:
            raise ValueError("Provide distance_mm or distance_measurements")
        # If empty and full distances known, simple linear mapping
        if self.empty_distance_mm is not None and self.full_distance_mm is not None:
            # distance reduces when coffee level rises, so fill ratio increases as distance decreases
            t = (self.empty_distance_mm - distance_mm) / (self.empty_distance_mm - self.full_distance_mm + 1e-9)
            t = float(np.clip(t, 0.0, 1.0))
            height_mm = None
            if self.pot_height_mm is not None:
                height_mm = t * self.pot_height_mm
            return t, height_mm
        # If pot height and distance to rim known: h_coffee = pot_height - (distance - distance_to_rim)
        if self.pot_height_mm is not None:
            # require user to give distance to liquid surface relative to same mount
            # we can't compute absolute without further calibration, so return None for now
            # fallback: return None but give the raw distance
            return None
        # else return raw distance as a negative indicator
        return None


# --- Fusion helper ---

def fuse_estimates(estimates, weights=None):
    """Fuse multiple fill-ratio estimates (each in [0,1]) using weighted average.
    estimates: dict of {name: (fill_ratio or None)}
    weights: optional dict of {name: weight}
    Returns fused_fill in [0,1] and dictionary of used estimates.
    """
    valid = {k: v for k,v in estimates.items() if v is not None}
    if not valid:
        return None, {}
    if weights is None:
        weights = {k: 1.0 for k in valid}
    total_w = 0.0
    accum = 0.0
    used = {}
    for k,v in valid.items():
        w = float(weights.get(k, 1.0))
        accum += w * float(v)
        total_w += w
        used[k] = (v, w)
    fused = float(accum / total_w)
    return fused, used


# ------------------
# Three short examples (as requested):
# 1) Mask segmenter
# 2) Classical CV
# 3) ToF mapping
# ------------------

if __name__ == "__main__":
    image = "./sample_coffee.jpg"
    # Example 1: mask segmenter
    msm = MaskSegmenterModel()
    try:
        r1 = msm.estimate_fill(image)
        print("MaskSegmenter fill:", r1)
    except Exception as e:
        print("MaskSegmenter failed:", e)

    # Example 2: classical CV
    ccm = ContourColorModel()
    try:
        r2 = ccm.estimate_fill(image)
        print("ContourColor fill:", r2)
    except Exception as e:
        print("ContourColor failed:", e)

    # Example 3: ToF mapping (simulate measurement)
    tof = ToFModel(pot_height_mm=200, empty_distance_mm=400, full_distance_mm=220)
    simulated_distance = 310  # mm measured from mount
    r3, height_mm = tof.estimate_fill(distance_mm=simulated_distance)
    print("ToF fill:", r3, "height_mm:", height_mm)
