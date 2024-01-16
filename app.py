from flask import Flask, jsonify, request

from doctr.file_utils import is_tf_available
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import cv2
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


import os
os.environ["USE_TORCH"] = "1"

from PIL import Image
import io
import base64

@app.route('/api/remove-text-from-image', methods=['POST'])
def removeTextFromImage():

    model = ocr_predictor('linknet_resnet50', 'crnn_vgg16_bn', pretrained=True)
    data = request.get_json()
    image_base64_list = data['images']

    resultList = []
    for image_base64 in image_base64_list:

        # header, encoded = image_base64.split(",", 1)
        
        image_bytes = base64.b64decode(image_base64)
        doc = DocumentFile.from_images(image_bytes)

        out = model(doc)

        result = ''
        for page in out.pages:
            result = page.show()

        resultList.append(result)

    print(resultList)

    data = {'result': resultList}
    return jsonify(data)

@app.route('/api/get-text-from-image', methods=['POST'])
def getTextFromImage():

    model = ocr_predictor('linknet_resnet50', 'crnn_vgg16_bn', pretrained=True)
    data = request.get_json()
    image_base64_list = data['images']

    resultList = []
    for image_base64 in image_base64_list:

        header, encoded = image_base64.split(",", 1)

        image_bytes = base64.b64decode(encoded)
        doc = DocumentFile.from_images(image_bytes)

        out = model(doc)

        output = out.render()

        resultList.append(output)

    data = {'result': resultList}
    return jsonify(data)

@app.route('/api/compare-image', methods=['POST'])
def compateImage():
    
    def compare_images(image1, image2):
        # Đọc ảnh
        img1 = image1
        img2 = image2

        # Chuyển đổi ảnh sang không gian màu HSV (Hue, Saturation, Value)
        hsv_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

        # Tính histogram của ảnh
        hist_img1 = cv2.calcHist([hsv_img1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist_img2 = cv2.calcHist([hsv_img2], [0, 1], None, [180, 256], [0, 180, 0, 256])

        # Chuẩn hóa histogram
        cv2.normalize(hist_img1, hist_img1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_img2, hist_img2, 0, 1, cv2.NORM_MINMAX)

        # Tính toán sự tương đồng giữa hai histogram
        similarity = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)

        return similarity

    # Đường dẫn của hai ảnh cần so sánh

    data = request.get_json()
    image_base64_list = data['images']

    image1_base64 = image_base64_list[0]  # raw data with base64 encoding
    image2_base64 = image_base64_list[1]  # raw data with base64 encoding

    decoded_data1 = base64.b64decode(image1_base64)
    np_data1 = np.fromstring(decoded_data1,np.uint8)
    image_path1 = cv2.imdecode(np_data1,cv2.IMREAD_UNCHANGED)

    decoded_data2 = base64.b64decode(image2_base64)
    np_data2 = np.fromstring(decoded_data2,np.uint8)
    image_path2 = cv2.imdecode(np_data2,cv2.IMREAD_UNCHANGED)

    # So sánh ảnh và in ra mức độ tương đồng
    similarity_score = compare_images(image_path1, image_path2) * 100

    # result = ''

    # if (similarity_score > 95):
    #     result = (f'Giống nhau, tương đồng giữa hai ảnh: {similarity_score} %')
    # else:
    #     result = (f'Khác nhau,tương đồng giữa hai ảnh: {similarity_score} %')

    data = {'result': similarity_score}
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5000')