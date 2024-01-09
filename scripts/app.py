from flask import Flask, jsonify

from doctr.file_utils import is_tf_available
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

app = Flask(__name__)

import os
os.environ["USE_TORCH"] = "1"

@app.route('/api/data', methods=['GET'])
def get_data():

    model = ocr_predictor('linknet_resnet50', 'crnn_vgg16_bn', pretrained=True)

    doc = DocumentFile.from_images('/Volumes/macOS/GNF-JAPAN/self-build/4.jpeg')

    out = model(doc)

    result = ''

    for page in out.pages:
        result = page.show()

    data = {'result': result}
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)