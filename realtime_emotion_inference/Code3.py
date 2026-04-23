import cv2
import numpy as np

# ----------- Camera calibration matrix and distortion coefficients -----------
# IMPORTANT: Replace these with your real calibration data for best results!
camera_matrix = np.array([[950, 0, 320], [0, 950, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))   # If you have calibration data, use actual values here.

# ----------- Set up ArUco marker dictionary and detection parameters -----------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()

def draw_ar_cube(image, corners, rvec, tvec):
    # Define 3D cube points in marker coordinate system
    marker_size = 0.05  # Marker side length in meters
    cube_points = marker_size * np.float32([
        [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
        [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]
    ])
    imgpts, _ = cv2.projectPoints(cube_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw base
    image = cv2.drawContours(image, [imgpts[:4]], -1, (0,255,0), 2)
    # Draw pillars
    for i, j in zip(range(4), range(4,8)):
        image = cv2.line(image, tuple(imgpts[i]), tuple(imgpts[j]), (255,0,0), 2)
    # Draw top
    image = cv2.drawContours(image, [imgpts[4:]], -1, (0,0,255), 2)
    return image

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        for corner in corners:
            # Estimate pose for each marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, 0.05, camera_matrix, dist_coeffs)
            cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
            frame = draw_ar_cube(frame, corner, rvec, tvec)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    else:
        cv2.putText(frame, "Show an ArUco marker to the camera.", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Augmented Reality with Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
