import os

from chest_classifier.pipeline.inference import InferencePipeline
from chest_classifier.utils.common import decodeImg

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

# Initialize flask app
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.image_name = "inputImage.jpg"
        self.classifier = InferencePipeline(self.image)

# Default route: main page
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['GET', 'POST'])
@cross_origin
def trainRoute():
    os.system("python main.py")
    # os.system("dvc repro")
    return "Training done successfully!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImg(image, clApp.image_name)
    result = clApp.classifier.predict()
    return jsonify(result)

if __name__ == '__main__':
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080) # for aws
