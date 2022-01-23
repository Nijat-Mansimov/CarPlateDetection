import os
import cv2
import matplotlib.pyplot as plt
from findPlate import findCarPlate

data = os.listdir("Cars")

for image_url in data:
    img = cv2.imread("Cars/" + image_url)
    img = cv2.resize(img, (500, 500))
    plate = findCarPlate(img)  # return -> x,y,w,h
    x, y, w, h, = plate

    # Artiq burada nomreni tapiriq ve o hisseni ayiririq yeni kesirik
    if w > h:
        plate_bgr = img[y:y + h, x:x + w].copy()

    else:
        plate_bgr = img[y:y + w, x:x + h].copy()

    img = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
