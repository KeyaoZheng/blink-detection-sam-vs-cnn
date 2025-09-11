
# data_utils/mp_mask.py
# Optional MediaPipe Face Mesh based eyelid fissure mask.
# If mediapipe is not installed or landmarks fail, returns None (caller should fallback).
import numpy as np

def mediapipe_eye_mask(img_rgb):
    '''
    img_rgb: HxWx3 uint8 RGB
    returns: mask float32 in {0,1} of shape (H,W), or None if failed
    '''
    try:
        import mediapipe as mp
    except Exception:
        return None
    H, W = img_rgb.shape[:2]
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as fm:
        res = fm.process(img_rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        pts = [(int(p.x * W + 0.5), int(p.y * H + 0.5)) for p in lm.landmark]
    # eyelid landmark indices (standard MP mesh)
    import cv2
    mask = np.zeros((H, W), np.uint8)
    try:
        # Left eye polygons (upper and lower), indices from MP documentation
        left_upper_idx = [159, 158, 157, 173, 133]
        left_lower_idx = [145, 144, 163, 7, 33]
        right_upper_idx = [386, 385, 384, 398, 263]
        right_lower_idx = [374, 380, 381, 382, 362]

        def poly_from_indices(indices):
            return np.array([pts[i] for i in indices], dtype=np.int32)

        left_upper = poly_from_indices(left_upper_idx)
        left_lower = poly_from_indices(left_lower_idx)
        right_upper = poly_from_indices(right_upper_idx)
        right_lower = poly_from_indices(right_lower_idx)

        for up, lo in [(left_upper, left_lower), (right_upper, right_lower)]:
            poly = np.vstack([up, lo[::-1]])
            cv2.fillPoly(mask, [poly], 255)
    except Exception:
        return None
    return (mask>127).astype(np.float32)
