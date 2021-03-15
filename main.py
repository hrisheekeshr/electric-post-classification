from detect_pole import detect_objects
from crop import crop
from predict import predict
from split import long_slice
from plotter import show_image

predicted_images = []
predicted_classes = []

image , box = detect_objects("pole_photos/CT_EPT_Pole-Transformer_740x530-2021-02-06T08:39:29.953Z.jpg", "first", .25)

cropped_image = crop(image,box)

image_slices = long_slice(cropped_image, 200)

for index,image in enumerate(image_slices):
    predicted_class, predicted_image = predict(image)
    predicted_classes.append(predicted_class)
    predicted_images.append(predicted_image)

print("Votes for Wood" , predicted_classes.count('wood'))
print("Votes for Metal", predicted_classes.count('metal'))

