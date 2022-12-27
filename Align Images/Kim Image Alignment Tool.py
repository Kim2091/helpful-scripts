import cv2
import numpy as np
import os
import logging
import argparse
import time

# Set up logging
logging.basicConfig(level=logging.INFO)

# Prompt the user for the file paths and the alignment mode
parser = argparse.ArgumentParser()
parser.add_argument("--folder1", "-f1", type=str, required=True, help="Path to the first folder")
parser.add_argument("--folder2", "-f2", type=str, required=True, help="Path to the second folder")
parser.add_argument("--output", "-o", type=str, required=True, help="Path to the output folder")

# Parse arguments
args = parser.parse_args()
folder1_path = args.folder1
folder2_path = args.folder2
output_path = args.output

# Normalize the file paths to parse backslashes
folder1_path = os.path.normpath(folder1_path)
folder2_path = os.path.normpath(folder2_path)
output_path = os.path.normpath(output_path)

# Get the list of images in both folders
folder1_images = os.listdir(folder1_path)
folder2_images = os.listdir(folder2_path)


def align_image(image1, image2):
    # Get the size of the first image
    size = (image1.shape[1], image1.shape[0])

    # Resize the second image to the same size as the first image
    image2 = cv2.resize(image2, size)
    
    # Convert the images to grayscale
    try:
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        logging.error("Error converting images to grayscale: %s", e)
        return

    # Find the keypoints and descriptors for both images
        # Use SIFT for keypoint detection and description
    try:
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image1_gray, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image2_gray, None)
    except Exception as e:
        logging.error("Error finding keypoints and descriptors: %s", e)
        return
    # Used to debug
    #print(f'Number of keypoints in image1: {len(keypoints1)}')
    #print(f'Number of keypoints in image2: {len(keypoints2)}')

    # Match the keypoints
        # Use FLANN for keypoint matching
    try:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = [m[0] for m in flann.knnMatch(descriptors1, descriptors2, k=2)]
    except Exception as e:
        logging.error("Error matching keypoints: %s", e)
        return

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Estimate the transformation using the top n matches
    n = 20  # Number of top matches to use
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches[:n]]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches[:n]]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    # Print the homography matrix - Used to debug
    #print(f'Homography Matrix: \n{M}')

    # Warp the image
    h, w = image1.shape[:2]
    aligned_image = cv2.warpPerspective(image1, M, (w, h))

    # Draw keypoints on image1
    image1_keypoints = cv2.drawKeypoints(aligned_image, keypoints1, outImage=np.array([]), color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Draw keypoints on image2
    image2_keypoints = cv2.drawKeypoints(image2, keypoints2, outImage=np.array([]), color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Save the images with keypoints drawn on them to file
    cv2.imwrite("image1_keypoints.png", image1_keypoints)
    cv2.imwrite("image2_keypoints.png", image2_keypoints)

    return aligned_image
	
# Iterate through the files in both folders
for file in folder1_images:
    # Check if the file exists in both folders
    if file in folder2_images:
        # If the file exists in both folders, align the image in folder1 with the image in folder2
        start_time = time.time()
        image1 = cv2.imread(os.path.join(folder1_path, file), cv2.IMREAD_UNCHANGED)
        image2 = cv2.imread(os.path.join(folder2_path, file), cv2.IMREAD_UNCHANGED)
        aligned_image = align_image(image1, image2)
        elapsed_time = time.time() - start_time
        
        # Save Files
        output_file = os.path.join(output_path, file)
        cv2.imwrite(output_file, aligned_image)

        # Print time taken per image
        print(f'Processed {file} in {elapsed_time:.2f} seconds')

print('Done!')
