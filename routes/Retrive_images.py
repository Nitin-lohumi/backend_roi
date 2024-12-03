from flask import Flask, request, jsonify,Blueprint,current_app
from flask_cors import CORS
from RoI_functions import get_matched_images_paths_with_ranking
import base64
import cv2
import os
dataset_path = '../Frontend/public/Animals'
UPLOAD_FOLDER = '../Frontend/public/QueryImages'
image_routes = Blueprint('image_routes', __name__)
CORS(image_routes)
@image_routes.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        image_data =data['imagelink']
        rois=data['rois']        
        if not image_data or not rois:
            return jsonify({'error': 'Missing image or rois'}), 400
        if image_data.startswith('data:image/png;base64,'):
            image_data = image_data.replace('data:image/png;base64,', '')
        if image_data.startswith('data:image/jpeg;base64,'):
            image_data = image_data.replace('data:image/jpeg;base64,', '')
        if image_data.startswith('data:image/jpg;base64,'):
            image_data = image_data.replace('data:image/jpg;base64,', '')
        x=rois['x']
        y=rois['y']
        w=rois['width']
        h=rois['height']
        image_data = image_data + '=' * (4 - len(image_data) % 4)     
        image_binary = base64.b64decode(image_data)
        image_filename = 'Image_Query.jpg' 
        image_path = os.path.join(UPLOAD_FOLDER, image_filename).replace("\\", "/")
        with open(image_path, 'wb') as image_file:
            image_file.write(image_binary)

        query_image = cv2.imread(image_path)
        roi_cropped = query_image[y:y+h, x:x+w] 
        if w==0:
            roi_cropped=query_image
        matched_image_paths = get_matched_images_paths_with_ranking(roi_cropped, dataset_path)
        return jsonify({"data":{"Matched_images":matched_image_paths}})

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
