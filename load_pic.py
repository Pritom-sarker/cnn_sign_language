from PIL import Image
import numpy as np



def get_image(image_path):
    """Get a numpy array of an image so that one can access values[x][y]."""
    image = Image.open(image_path, 'r')
    width, height = image.size
    pixel_values = list(image.getdata())
    if image.mode == 'RGB':
        channels = 3
    elif image.mode == 'L':
        channels = 1
    else:
        print("Unknown mode: %s" % image.mode)
        return None
    pixel_values = np.array(pixel_values)
    #.reshape((width, height, channels))
    return pixel_values

li=[]

for i in range(0,9):
    for j in range(1,20):
        if j >9:
            name = '''G:\Aminul sir projects\sign language\leapGestRecog\\0{}\\01_palm\\frame_0{}_01_00{}.png'''.format(
                i, i, j)
        else:
            name='''G:\Aminul sir projects\sign language\leapGestRecog\\0{}\\01_palm\\frame_0{}_01_000{}.png'''.format(i,i,j)
        print(name)
        li.append(get_image(name))


        break

print(len(li))

import pickle
from sklearn.externals import joblib
arr=np.array(li)

joblib.dump(arr,'data/01.pkl')


