import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class InferencePipeline:
    def __init__(self, image):
        self.image = image

    def predict(self):
        #model = load_model("artifacts/training/model.h5") # Local
        model = load_model(os.path.join("model", "model.h5")) # Deployment

        img = self.image
        test_image = image.load_img(img, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Normal'
            return [{'image' : prediction}]
        else:
            prediction = 'Adenocarcinoma Cancer'
            return [{'image': prediction}]
