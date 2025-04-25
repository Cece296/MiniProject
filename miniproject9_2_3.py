import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------- USER SETTINGS --------
FEATURE_TYPE = 'SIFT'      # 'SIFT' or 'ORB'
FRAME1_PATH = 'extractedframes/frame_0001.jpg'
FRAME2_PATH = 'extractedframes/frame_0002.jpg'

# Camera calibration for 3840x2160 Phantom 4 Pro
K = np.array([
    [2559,    0, 1919],
    [   0, 2159, 1281],
    [   0,    0,    1]
])
# --------------------------------

def epipolar_line_distances(pts1, pts2, F):
    N = pts1.shape[0]
    pts1_h = np.hstack([pts1, np.ones((N, 1))])
    pts2_h = np.hstack([pts2, np.ones((N, 1))])
    lines2 = (F @ pts1_h.T).T
    a, b, c = lines2[:,0], lines2[:,1], lines2[:,2]
    x2, y2 = pts2[:,0], pts2[:,1]
    dists = np.abs(a*x2 + b*y2 + c) / np.sqrt(a**2 + b**2)
    return dists

# 1. Load frames as grayscale
frame1 = cv2.imread(FRAME1_PATH, cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread(FRAME2_PATH, cv2.IMREAD_GRAYSCALE)
assert frame1 is not None, f"Could not load {FRAME1_PATH}"
assert frame2 is not None, f"Could not load {FRAME2_PATH}"

# 2. Feature detection
if FEATURE_TYPE.upper() == 'SIFT':
    detector = cv2.SIFT_create()
else:
    detector = cv2.ORB_create()
keypoints1, descriptors1 = detector.detectAndCompute(frame1, None)
keypoints2, descriptors2 = detector.detectAndCompute(frame2, None)

print(f"Frame 1: {len(keypoints1)} keypoints. Frame 2: {len(keypoints2)} keypoints.")

# 3. Feature matching
if descriptors1.dtype == np.float32:
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
else:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
print(f"Found {len(matches)} matches.")

# 4. Matched points arrays
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

# 5. Fundamental matrix
F, Fmask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
# 6. Essential matrix
E, Emask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# 7. Epipolar distances (using F, for raw pixel points)
epi_dists = epipolar_line_distances(pts1, pts2, F)
print(f"\nEpipolar distances (pixels):")
print(f"  Mean: {np.mean(epi_dists):.4f}")
print(f"  Std:  {np.std(epi_dists):.4f}")
print(f"  Min:  {np.min(epi_dists):.4f}")
print(f"  Max:  {np.max(epi_dists):.4f}")

# 8. Visualize inlier matches (by F)
inlier_matches = [matches[i] for i in range(len(matches)) if Fmask[i]]
img_matches = cv2.drawMatches(
    frame1, keypoints1, frame2, keypoints2, inlier_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(16, 8))
plt.title("Inlier Feature Matches (Fundamental Matrix)")
plt.imshow(img_matches)
plt.axis('off')
plt.tight_layout()
plt.show()
