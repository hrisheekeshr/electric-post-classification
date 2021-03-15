from __future__ import division
from PIL import Image
import math
import os

def long_slice(img,  slice_size):
    """slice an image into parts slice_size tall"""
    width, height = img.size
    upper = 0
    left = 0
    slices = int(math.ceil(height/slice_size))
    images_list = []
    count = 1
    for slice in range(slices):
        #if we are at the end, set the lower bound to be the bottom of the image
        if count == slices:
            lower = height
        else:
            lower = int(count * slice_size)

        bbox = (left, upper, width, lower)
        working_slice = img.crop(bbox)
        upper += slice_size
        #save the slice
        images_list.append(working_slice)

        outdir = 'slices'
        working_slice.save(os.path.join(outdir, "slice_" + str(slice) + "_" + str(count)+".png"))
        count +=1
    return images_list

# if __name__ == '__main__':
#     long_slice("out.jpg", 200)