import matplotlib.pyplot as plt
def show_image(image, title):

    plt.axis('off')
    _ = plt.title("Prediction: " + title)
    plt.imshow(image)