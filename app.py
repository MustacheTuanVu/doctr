from flask import Flask, jsonify, request

from doctr.file_utils import is_tf_available
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

app = Flask(__name__)

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
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5000')