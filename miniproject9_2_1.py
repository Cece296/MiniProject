import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---- USER SETTINGS ----
FEATURE_TYPE = 'ORB'    # Change to 'SIFT' if you want SIFT
FRAME1_PATH = 'extractedframes/frame_0001.jpg'
FRAME2_PATH = 'extractedframes/frame_0002.jpg'
# -----------------------

# Load frames as grayscale
frame1 = cv2.imread(FRAME1_PATH, cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread(FRAME2_PATH, cv2.IMREAD_GRAYSCALE)
assert frame1 is not None, f"Could not load {FRAME1_PATH}"
assert frame2 is not None, f"Could not load {FRAME2_PATH}"

# Select feature detector
if FEATURE_TYPE.upper() == 'SIFT':
    detector = cv2.SIFT_create()
elif FEATURE_TYPE.upper() == 'ORB':
    detector = cv2.ORB_create()
else:
    raise ValueError("Unknown feature type. Use 'SIFT' or 'ORB'.")

# Extract features
keypoints1, descriptors1 = detector.detectAndCompute(frame1, None)
keypoints2, descriptors2 = detector.detectAndCompute(frame2, None)

print(f"Frame 1: {len(keypoints1)} keypoints found.")
print(f"Frame 2: {len(keypoints2)} keypoints found.")

# (Optional) Save keypoints and descriptors for later use
# np.save('keypoints1.npy', np.array([kp.pt for kp in keypoints1]))
# np.save('descriptors1.npy', descriptors1)
# np.save('keypoints2.npy', np.array([kp.pt for kp in keypoints2]))
# np.save('descriptors2.npy', descriptors2)

# Draw keypoints for visualization
img_with_kp1 = cv2.drawKeypoints(frame1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_with_kp2 = cv2.drawKeypoints(frame2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title('Frame 1 Keypoints')
plt.imshow(img_with_kp1, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Frame 2 Keypoints')
plt.imshow(img_with_kp2, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
