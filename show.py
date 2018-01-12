from PIL import Image
import numpy as np

i32 = np.load('img.npy')
i8 = i32.astype(np.uint8)
img = Image.fromarray(i8)
img.show()
