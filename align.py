import cv2
import numpy as np
import sys


def align_to_master(master, new_img):
    """Align a new drone image to the master reference using SIFT + FLANN.

    Returns (result_dict) where result_dict contains:
        aligned: the warped image (or original if failed)
        homography: the 3x3 matrix (or None if failed)
        keypoints_master: number of SIFT keypoints in master
        keypoints_live: number of SIFT keypoints in new image
        good_matches: number of matches after Lowe's ratio test
        inliers: number of RANSAC inliers (0 if failed)
    """
    gray1 = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    result = {
        "aligned": new_img,
        "homography": None,
        "keypoints_master": len(kp1) if kp1 else 0,
        "keypoints_live": len(kp2) if kp2 else 0,
        "good_matches": 0,
        "inliers": 0,
    }

    if des1 is None or des2 is None:
        return result

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    result["good_matches"] = len(good_matches)

    if len(good_matches) < 10:
        return result

    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        return result

    result["inliers"] = int(mask.ravel().sum())
    result["homography"] = H
    result["aligned"] = cv2.warpPerspective(new_img, H, (master.shape[1], master.shape[0]))

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
