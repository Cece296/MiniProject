import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------- USER SETTINGS --------
FEATURE_TYPE = 'ORB'      # 'SIFT' or 'ORB'
FRAME1_PATH = 'extractedframes/frame_0001.jpg'
FRAME2_PATH = 'extractedframes/frame_0002.jpg'
# Camera calibration matrix (fx, fy, cx, cy)
# Replace these with your actual camera calibration!

K = np.array([
    [2559,    0, 1919],
    [   0, 2159, 1281],
    [   0,    0,    1]
])
# --------------------------------

# 1. Load frames as grayscale
frame1 = cv2.imread(FRAME1_PATH, cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread(FRAME2_PATH, cv2.IMREAD_GRAYSCALE)
assert frame1 is not None, f"Could not load {FRAME1_PATH}"
assert frame2 is not None, f"Could not load {FRAME2_PATH}"

# 2. Create feature detector
if FEATURE_TYPE.upper() == 'SIFT':
    detector = cv2.SIFT_create()
elif FEATURE_TYPE.upper() == 'ORB':
    detector = cv2.ORB_create()
else:
    raise ValueError("Unknown feature type. Use 'SIFT' or 'ORB'.")

# 3. Detect and compute
keypoints1, descriptors1 = detector.detectAndCompute(frame1, None)
keypoints2, descriptors2 = detector.detectAndCompute(frame2, None)

print(f"Frame 1: {len(keypoints1)} keypoints. Frame 2: {len(keypoints2)} keypoints.")

# 4. Match features
if descriptors1.dtype == np.float32:
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)   # SIFT
else:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # ORB

matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

print(f"Found {len(matches)} matches.")

# 5. Extract matched points
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

# 6. Estimate essential matrix
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]

print(f"Essential matrix:\n{E}")
print(f"Inlier matches after RANSAC: {np.sum(mask)}")

# 7. Visualize inlier matches
img_matches = cv2.drawMatches(
    frame1, keypoints1, frame2, keypoints2, inlier_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(16, 8))
plt.title("Inlier Feature Matches (for Essential Matrix)")
plt.imshow(img_matches)
plt.axis('off')
plt.tight_layout()
plt.show()
