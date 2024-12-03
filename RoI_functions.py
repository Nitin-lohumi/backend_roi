import os
import numpy as np
import cv2
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import cosine_similarity
import requests
def load_images_from_folder(folder):
    images = []
    
    filenames = []
    
    for filename in os.listdir(folder):
        
        img = cv2.imread(os.path.join(folder, filename))
        
        if img is not None:
            
            images.append(img)
            
            filenames.append(filename)
            
    return images, filenames



# def select_roi(image):
#     roi = cv2.selectROI("Select ROI", image, False, False)
#     cv2.destroyWindow("Select ROI")
#     return roi




def get_matched_images_paths_with_ranking(query_image, image_folder, threshold=250, min_matches=20):
    images, filenames = load_images_from_folder(image_folder)
    kp1, des1 = compute_sift_features(query_image)

    matched_images = []
    total_matches = 0
    matched_photos_count = 0

    print("\nMatching Results:")
    print("-" * 40)

    for img, filename in zip(images, filenames):

        kp2, des2 = compute_sift_features(img)
        good_matches = match_descriptors(des1, des2, threshold)

        print(f"File: {filename}, Good Matches: {len(good_matches)}")

        total_matches += len(good_matches)
        if len(good_matches) >= min_matches:
            matched_photos_count += 1
            # matched_images.append((len(good_matches), os.path.join(image_folder, filename).replace("\\", "/")))
            matched_images.append((len(good_matches), filename))
    matched_images = sorted(matched_images, key=lambda x: x[0], reverse=True)
    matched_image_paths = [image[1] for image in matched_images]
    print("\nSummary:")
    print(f"Total images processed: {len(images)}")
    print(f"Number of matched photos: {matched_photos_count}")
    print(f"Total number of good matches across all images: {total_matches}")
    print("\nTop Matches:")
    for i, (filename, match_count) in enumerate(matched_images, start=1):
        print(f"{i}. {filename} - Matches: {match_count}")
    return matched_image_paths



def compute_sift_features(image):
    sift = cv2.SIFT_create()  
    
    keypoints, descriptors = sift.detectAndCompute(image, None) 
    
    return keypoints, descriptors




def match_descriptors(des1, des2, threshold):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)   
    matches = bf.match(des1, des2)   
    matches = sorted(matches, key=lambda x: x.distance) 
    good_matches = [match for match in matches if match.distance < threshold]   
    return good_matches




# query_image = cv2.imread("Animals/cheetah-171217__340.jpg")


# roi  = select_roi(query_image)
# x, y, w, h = roi
# roi_cropped = query_image[y:y+h, x:x+w]
# plt.imshow(cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2RGB))
# plt.title("Selected ROI")
# plt.show()


# print(roi)

# images, filenames = load_images_from_folder(dataset_path)

# print(f"Loaded {len(images)} images.")

# query_features = extract_features(roi_cropped)


# dataset_features = []
# for img in images:
#     descriptors = extract_features(img)
#     dataset_features.append(descriptors)


# matches = match_features(query_features, dataset_features)

# print(len(matches))

# match_scores = [len(matches) for matches_list in matches]


# num_images = len(images)
# num_matches = len(match_scores)

# top_n = 10
# sorted_indices = np.argsort(match_scores[:num_images])[::-1]
# print(sorted_indices)




# for i in range(min(top_n, len(sorted_indices))):  # Ensure we don't go out of range
#     idx = sorted_indices[i]
#     if idx < num_images:
#         print(f"Image {idx}: {match_scores[idx]} matches")
#         img_to_show = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
#         plt.figure(figsize=(2, 10))
#         plt.imshow(img_to_show)
#         plt.title(f"Top {i+1} Match - Image {idx} | Matches: {match_scores[idx]}")
#         # plt.axis('off')
#         plt.show()
#     else:
#         print(f"Invalid index {idx} for the images list. Total images: {num_images}")



# def load_online_image(url):
#     try:
#         # Fetch the image from the URL
#         response = requests.get(url, stream=True)
#         response.raise_for_status()  # Raise an error for bad status codes
#         # Convert the response content to a NumPy array
#         image_data = np.asarray(bytearray(response.content), dtype=np.uint8)
#         # Decode the image
#         img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
#         if img is None:
#             print(f"Failed to decode the image from URL: {url}")
#         else:
#             print(f"Successfully loaded the image from URL: {url}")
#         return img
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching the image from URL: {e}")
#         return None




