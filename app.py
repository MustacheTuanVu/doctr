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

@app.route('/api/data', methods=['POST'])
def get_data():

    data = request.get_json()

    image_base64 = data['image']

    model = ocr_predictor('linknet_resnet50', 'crnn_vgg16_bn', pretrained=True)

    image_bytes = base64.b64decode(image_base64)

    # Get the bytes
    # bytes_data = img_bytes.getvalue()

    doc = DocumentFile.from_images(image_bytes)

    out = model(doc)

    result = ''

    for page in out.pages:
        result = page.show()

    data = {'result': result}
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5000')