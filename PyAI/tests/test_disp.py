# importing image object from PIL
import math
from PIL import Image, ImageDraw
  
w, h = 220, 190
shape = [(40, 40), (w - 10, h - 10)]
  
# creating new Image object
img = Image.new("RGB", (w, h))
  
# create line image
img1 = ImageDraw.Draw(img)  
img1.line(shape, fill ="red", width = 0)
img.show()
