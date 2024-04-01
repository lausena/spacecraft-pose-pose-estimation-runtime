import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SIFTFeatureTracker:
    def __init__(self):
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()
        self.poses = []
    
    def detect_and_describe(self, image):
        """
        Detect keypoints and compute descriptors for a given image.
        """
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """
        Match descriptors between two images using FLANN-based matcher.
        """
        # Initialize and configure matcher
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        return good_matches

    def estimate_motion(self, matched_kps1, matched_kps2, K):
        """
        Estimate camera pose using matched keypoints and camera intrinsic matrix.
        """
        points1 = np.float32([kp.pt for kp in matched_kps1])
        points2 = np.float32([kp.pt for kp in matched_kps2])
        try:
            E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        except:
            return None, None
    
        
        try:
            _, R, t, _ = cv2.recoverPose(E, points1, points2, K, mask)
            return R, t
        except cv2.error as e:
            print(f"OpenCV error encountered: {e}")
            return None, None
        
        
    
    def process_pair(self, image1, image2, K):
        """
        Process a pair of images: detect and match features, then estimate motion.
        """
        kps1, desc1 = self.detect_and_describe(image1)
        kps2, desc2 = self.detect_and_describe(image2)
        
        
        matches = self.match_features(desc1, desc2)
        
        matched_kps1 = [kps1[m.queryIdx] for m in matches]
        matched_kps2 = [kps2[m.trainIdx] for m in matches]
        
        R, t = self.estimate_motion(matched_kps1, matched_kps2, K)
        self.poses.append((R, t))
        return R, t
    def visualize_poses(self):
        """
        Visualized set of poses per image sequence
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        origin = np.array([0, 0, 0, 1])
        for R, t in self.poses:
            # Transform origin and plot
            cam_pos = -R.T @ t
            ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], R[0, 0], R[1, 0], R[2, 0], length=0.1, color='r')
            ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], R[0, 1], R[1, 1], R[2, 1], length=0.1, color='g')
            ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], R[0, 2], R[1, 2], R[2, 2], length=0.1, color='b')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def r_to_q(self, R):
        """
        Convert a rotation matrix to a quaternion.
        
        Parameters:
        R (np.array): A 3x3 rotation matrix.
        
        Returns:
        np.array: Quaternion [w, x, y, z].
        """
        T = np.trace(R)
        
        if T > 0.0:
            S = np.sqrt(T + 1.0) * 2  # S=4*qw 
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S 
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        
        return np.array([qw, qx, qy, qz])

# Example usage
if __name__ == "__main__":
    # Load images (grayscale)
    image1 = cv2.imread('path_to_your_first_image.jpg', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('path_to_your_second_image.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Initialize camera intrinsic matrix (example values)
    K = np.array([[700, 0, 320],
                  [0, 700, 240],
                  [0, 0, 1]], dtype=float)
    
    # Initialize the tracker and process the images
    tracker = SIFTFeatureTracker()
    R, t = tracker.process_pair(image1, image2, K)
    
    print("Estimated Rotation Matrix:\n", R)
    print("Estimated Translation Vector:\n", t)

