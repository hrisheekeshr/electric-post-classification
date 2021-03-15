from PIL import Image
import numpy
from detect_pole import detect_objects

# image = Image.open('pole_photos/CT_EPT_Pole-Transformer_740x530-2021-02-06T08:39:29.953Z.jpg')
# box=[0.104051828, 0.654694676, 1.0, 0.756162047]

def crop(image,box):
        arr = numpy.array(image)
        im_width, im_height = arr.shape[1], arr.shape[0]
        xmin  = box[1]
        ymin  = box[0]
        xmax  = box[3]
        ymax  = box[2]
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        a,b,c,d = int(left) , int(right) , int(top) ,int(bottom)
        arr = arr[c:d,a:b]
        cropped = Image.fromarray(arr)
        cropped.show()
        # cropped.save('out.jpg')
        return cropped

