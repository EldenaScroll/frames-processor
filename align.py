import cv2
import numpy as np
import sys


def precompute_master(master_img):
    """Pre-calculate SIFT keypoints and descriptors for a master image."""
    gray = cv2.cvtColor(master_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def align_to_master(master, new_img, master_kp=None, master_des=None):
    """Align a new drone image to the master reference using SIFT + FLANN with robust fallbacks.
    
    Args:
        master: the master image BGR array
        new_img: the live drone image BGR array
        master_kp: (optional) precompute_master keypoints
        master_des: (optional) precompute_master descriptors
    """
    gray1 = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    def attempt_alignment(kp1, des1, kp2, des2, ratio=0.7, ransac_thresh=5.0, use_flann=True):
        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            return None, 0, 0
            
        if use_flann:
            index_params = dict(algorithm=1, trees=5) 
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            
        try:
            matches = matcher.knnMatch(des1, des2, k=2)
        except Exception:
            return None, 0, 0

        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 10:
            return None, len(good_matches), 0

        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

        if H is None:
            return None, len(good_matches), 0

        inliers = int(mask.ravel().sum())
        return H, len(good_matches), inliers

    sift = cv2.SIFT_create()
    
    if master_kp is None or master_des is None:
        kp1, des1 = sift.detectAndCompute(gray1, None)
    else:
        kp1, des1 = master_kp, master_des
        
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    result = {
        "aligned": new_img,
        "homography": None,
        "keypoints_master": len(kp1) if kp1 else 0,
        "keypoints_live": len(kp2) if kp2 else 0,
        "good_matches": 0,
        "inliers": 0,
        "fallback_used": "None"
    }

    H, good, inliers = attempt_alignment(kp1, des1, kp2, des2, ratio=0.7)
    
    if H is None or inliers < 10:
        print("⚠️ Standard SIFT alignment failed, activating Fallback 1: Relaxed Ratio Test (0.8)")
        H, good, inliers = attempt_alignment(kp1, des1, kp2, des2, ratio=0.8)
        result["fallback_used"] = "Relaxed Ratio (0.8)"
        
    if H is None or inliers < 10:
        print("⚠️ Relaxed SIFT failed, activating Fallback 2: CLAHE Enhanced SIFT")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray1_clahe = clahe.apply(gray1)
        gray2_clahe = clahe.apply(gray2)
        
        kp1_c, des1_c = sift.detectAndCompute(gray1_clahe, None)
        kp2_c, des2_c = sift.detectAndCompute(gray2_clahe, None)
        
        H, good, inliers = attempt_alignment(kp1_c, des1_c, kp2_c, des2_c, ratio=0.75)
        result["fallback_used"] = "CLAHE + SIFT"
        
    if H is None or inliers < 10:
        print("⚠️ CLAHE SIFT failed, activating Fallback 3: ORB Matching")
        orb = cv2.ORB_create(nfeatures=5000)
        kp1_o, des1_o = orb.detectAndCompute(gray1, None)
        kp2_o, des2_o = orb.detectAndCompute(gray2, None)
        
        H, good, inliers = attempt_alignment(kp1_o, des1_o, kp2_o, des2_o, ratio=0.8, use_flann=False)
        result["fallback_used"] = "ORB Matcher"

    result["good_matches"] = good
    result["inliers"] = inliers

    if H is not None and inliers >= 10:
        result["homography"] = H
        result["aligned"] = cv2.warpPerspective(new_img, H, (master.shape[1], master.shape[0]))
    else:
        # All fallbacks failed
        result["homography"] = None
        result["aligned"] = new_img

    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python align.py <master_image> <test_image>")
        sys.exit(1)

    master = cv2.imread(sys.argv[1])
    if master is None:
        print(f"Error: Could not load master image: {sys.argv[1]}")
        sys.exit(1)

    test_img = cv2.imread(sys.argv[2])
    if test_img is None:
        print(f"Error: Could not load test image: {sys.argv[2]}")
        sys.exit(1)

    print(f"Master: {master.shape[1]}x{master.shape[0]}")
    print(f"Test:   {test_img.shape[1]}x{test_img.shape[0]}")

    result = align_to_master(master, test_img)

    print(f"SIFT keypoints: master={result['keypoints_master']}, live={result['keypoints_live']}")
    print(f"Good matches: {result['good_matches']}")
    print(f"RANSAC inliers: {result['inliers']}")

    if result["homography"] is not None:
        print("Alignment SUCCEEDED")
        cv2.imwrite("aligned_output.jpg", result["aligned"])
        print("Saved: aligned_output.jpg")
    else:
        print("Alignment FAILED — not enough matches or homography could not be computed")
