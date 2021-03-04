from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

IMAGE_SIZE = (299, 299)

labels = {
    1 : 'wood',
    0 : 'metal'
}

def predict(image_path):

    image = Image.open(image_path).resize(IMAGE_SIZE)
    image_np = np.array(image)/255.0

    new_model = tf.keras.models.load_model('models/saved_model')

    result = new_model.predict(image_np[np.newaxis, ...])
    predicted_class = np.argmax(result[0], axis=-1)

    plt.imshow(image)
    plt.axis('off')
    predicted_class_name = labels[predicted_class]
    _ = plt.title("Prediction: " + predicted_class_name.title())
    plt.show()

predict('test/metal.jpg')