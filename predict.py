from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

IMAGE_SIZE = (299, 299)

labels = {
    1 : 'wood',
    0 : 'metal'
}

new_model = tf.keras.models.load_model('models/saved_model_new')

def predict(image):

    image_resized = image.resize(IMAGE_SIZE)
    image_np = np.array(image_resized)/255.0

    result = new_model.predict(image_np[np.newaxis, ...])
    print(result)
    predicted_class = np.argmax(result[0], axis=-1)

    plt.imshow(image_resized)
    plt.axis('off')
    predicted_class_name = labels[predicted_class]
    _ = plt.title("Prediction: " + predicted_class_name.title())
    plt.show()
    return predicted_class_name, image

# image_path = 'test/wood.jpg'
# image = Image.open(image_path)
# predict(image)