import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------- USER SETTINGS --------
FEATURE_TYPE = 'ORB'      # 'SIFT' or 'ORB'
FRAME1_PATH = 'extractedframes/frame_0001.jpg'
FRAME2_PATH = 'extractedframes/frame_0002.jpg'

# Camera calibration for 3840x2160 Phantom 4 Pro
# --- Replace this camera matrix with the real one ---
K = np.array([
    [2676.1051, 0.0,       -35.24395],
    [0.0,       2676.1051, -279.5856],
    [0.0,       0.0,        1.0]
])

dist_coeffs = np.array([
    0.00979359,   # k1
   -0.02179405,   # k2
    0.00464436,   # p1
   -0.00456640,   # p2
    0.01777650    # k3
])


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

# 9. Recover relative camera pose from E and inliers
inliers = (Emask.ravel() == 1)
pts1_inliers = pts1[inliers]
pts2_inliers = pts2[inliers]


# cv2.recoverPose expects (N, 1, 2) shaped arrays
retval, R, t, mask_pose = cv2.recoverPose(E, pts1_inliers.reshape(-1,1,2), pts2_inliers.reshape(-1,1,2), K)
# Step 10: Triangulate 3D points from inlier matches

print("\n--------")
print(f"Using {pts1_inliers.shape[0]} inlier matches to estimate 3D point positions...")

# 1. Undistort the inlier points
pts1_undist = cv2.undistortPoints(pts1_inliers.reshape(-1, 1, 2), K, dist_coeffs).reshape(-1, 2)
pts2_undist = cv2.undistortPoints(pts2_inliers.reshape(-1, 1, 2), K, dist_coeffs).reshape(-1, 2)

print("Undistorted inlier points.")
print(f"First undistorted point from frame 1: {pts1_undist[0]}")
print(f"First undistorted point from frame 2: {pts2_undist[0]}")

# 2. Create projection matrices in normalized coordinates
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera 1 at origin
P2 = np.hstack((R, t))                         # Camera 2 pose
print("Constructed projection matrices P1 and P2.")

# 3. Perform triangulation
pts4D = cv2.triangulatePoints(P1, P2, pts1_undist.T, pts2_undist.T)

# 4. Convert to 3D Euclidean coordinates
pts3D = (pts4D[:3] / pts4D[3]).T  # Shape: (N, 3)
print(f"Triangulated {pts3D.shape[0]} 3D points.")

# 5. Show some of the triangulated points
print("\nFirst 5 triangulated 3D points (X, Y, Z):")
for i, pt in enumerate(pts3D[:5]):
    print(f"Point {i+1}: {pt}")

print("\nRecovered relative camera pose:")
print("Rotation matrix R:\n", R)
print("Translation vector t:\n", t)
print(f"Number of inlier correspondences used for pose: {retval}")
# Step 11: Create Map and add cameras, points, observations, and visualize

print("\n--- Step 11: Map Construction ---")

# 1. Define data classes
class Observation:
    def __init__(self, camera_id, feature_id, image_point):
        self.camera_id = camera_id
        self.feature_id = feature_id
        self.image_point = image_point

class TrackedPoint:
    def __init__(self, position):
        self.position = position
        self.observations = []

class TrackedCamera:
    def __init__(self, pose):
        self.pose = pose  # 4x4 matrix

class Map:
    def __init__(self):
        self.points = []
        self.cameras = []

# 2. Initialize map
map = Map()

# 3. Add camera 0 (origin)
pose0 = np.eye(4)
map.cameras.append(TrackedCamera(pose0))
print("Added Camera 0 at origin (identity pose).")

# 4. Add camera 1 (estimated pose)
pose1 = np.vstack([np.hstack([R, t]), [0, 0, 0, 1]])
map.cameras.append(TrackedCamera(pose1))
print("Added Camera 1 with recovered pose.")

# 5. Add points and their observations
for i, point in enumerate(pts3D):
    tracked_point = TrackedPoint(point)
    tracked_point.observations.append(Observation(0, i, pts1_inliers[i]))
    tracked_point.observations.append(Observation(1, i, pts2_inliers[i]))
    map.points.append(tracked_point)

print(f"Added {len(map.points)} triangulated points with 2 observations each.")

# 6. Visualize map
print("Visualizing cameras and 3D points...")



fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot 3D points
pts3D_np = np.array([pt.position for pt in map.points])
ax.scatter(pts3D_np[:, 0], pts3D_np[:, 1], pts3D_np[:, 2], s=3, c='blue', label='3D Points')

# Plot cameras
for i, cam in enumerate(map.cameras):
    cam_pos = cam.pose[:3, 3]
    ax.scatter(*cam_pos, c='red', s=50, marker='^', label=f'Camera {i}' if i == 0 else "")
    if i == 1:
        # draw camera orientation
        forward = cam.pose[:3, 2]
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                  forward[0], forward[1], forward[2],
                  length=0.5, color='green', label='Camera 1 Orientation')

ax.set_title('3D Map with Cameras and Triangulated Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.tight_layout()
plt.show()
# Step 12: Compute Reprojection Error

print("\n--- Step 12: Reprojection Error Evaluation ---")

def compute_reprojection_error(map, K, dist_coeffs):
    errors = []
    for pt in map.points:
        X = pt.position.reshape(1, 3).astype(np.float32)  # shape (1, 3)

        for obs in pt.observations:
            cam = map.cameras[obs.camera_id]
            R = cam.pose[:3, :3]
            t = cam.pose[:3, 3].reshape(3, 1)

            rvec, _ = cv2.Rodrigues(R)
            x_proj, _ = cv2.projectPoints(X, rvec, t, K, dist_coeffs)
            x_proj = x_proj.ravel()

            error = np.linalg.norm(obs.image_point - x_proj)
            errors.append(error)

    return errors

# Compute and report stats
errors = compute_reprojection_error(map, K, dist_coeffs)
print(f"Reprojection Error:")
print(f"  Mean: {np.mean(errors):.4f} pixels")
print(f"  Std:  {np.std(errors):.4f} pixels")
print(f"  Min:  {np.min(errors):.4f} pixels")
print(f"  Max:  {np.max(errors):.4f} pixels")

print("Visualizing 3D points color-coded by reprojection error...")

pts3D_np = np.array([pt.position for pt in map.points])

# Average reprojection error per 3D point
point_errors = np.array(errors).reshape(-1, 2).mean(axis=1)

# Normalize for colormap
errors_norm = (point_errors - point_errors.min()) / (point_errors.ptp() + 1e-8)
colors = plt.cm.viridis(errors_norm)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(pts3D_np[:, 0], pts3D_np[:, 1], pts3D_np[:, 2],
                s=5, c=colors, cmap='viridis')

ax.set_title('Triangulated Points Colored by Reprojection Error')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
cbar.set_label('Mean Reprojection Error (pixels)')
plt.tight_layout()
plt.show()




