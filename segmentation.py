import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from findPlate import findCarPlate

data = os.listdir("Cars")
name = data[0]

img = cv2.imread("Cars/" + name)
img = cv2.resize(img, (500, 500))

plate = findCarPlate(img)
x, y, w, h = plate

if w > h:
    plate_bgr = img[y:y + h, x:x + w].copy()

else:
    plate_bgr = img[y:y + w, x:x + h].copy()

plt.imshow(plate_bgr)
plt.show()

# Sekilde olan pikselliyi azaldiriq ki, icerisinde olan simvollari sece bilek
H, W = plate_bgr.shape[:2]
print("Orginal olcu: ", W, H)
H, W = H * 2, W * 2

plate_bgr = cv2.resize(plate_bgr, (W, H))

plt.imshow(plate_bgr)
plt.show()

PLATE_IMG = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

plt.title("Gray formatli sekil")
plt.imshow(PLATE_IMG, cmap="gray")
plt.show()

# Kicik parcalara bolerek sekli diqqete aliriq
# Pozitiv olanlari negativ, negativ olanlari pozitiv alaraq isimizi hell edirik
th_img = cv2.adaptiveThreshold(PLATE_IMG, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

plt.title("Thresholded olunmus sekil")
plt.imshow(th_img, cmap="gray")
plt.show()

# Lazimsiz hisseleri silirik / deleted unnecessary
kernel = np.ones((3, 3), np.uint8)
th_img = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, kernel, iterations=1)

plt.title("Lazimsiz hisseler olmadan")
plt.imshow(th_img, cmap="gray")
plt.show()

# Countrlari tapiriq ve iyeraxiyanin nece olacagini bildiririk
# Yalnizca xarici countrlari elde etmeliyik
cnt = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = cnt[0]

# Bize lazim olan reqem ve herfler countrlarin icinde ilk 10-luqda yer alacaq cunki onlardan elave basqa bir dene
# cercivedi Daha sonra cerciveni nezerden cixarmaliyiq

# Burada sahesi en boyuk olan simvollari kicikden boyuye dogru duzuruk cunki reqemlerin ve herflerin sahesi diger
# qirislardan boyukdur
cnt = sorted(cnt, key=cv2.contourArea, reverse=True)[:15]

for index, counter in enumerate(cnt):
    rect = cv2.minAreaRect(counter)  # sahesi en kicik olan kvadrat
    (x, y), (w, h), r = rect

    # Texminen bir herfin hundurluyu yeni eni umumi cervicenin uzunlugunun 4/1 hissesi qeder olur
    # Bunu nezere alaraq dusune bilerik ki bir herfin tutdugu kutunun saheside 200-den boyukdur
    control1 = max((w, h)) < W / 4
    control2 = w * h > 200

    if control1 and control2:
        print("Fiqur: ", x, y, w, h)

        # Tepe noqtelerini tapiriq
        box = cv2.boxPoints(rect)
        box = np.int64(box)

        minx = np.min(box[:, 0])
        miny = np.min(box[:, 1])
        maxx = np.max(box[:, 0])
        maxy = np.max(box[:, 1])

        # Reqem ve herfleri secerken margin olaraq 2 veririk
        focus = 2

        minx = max(0, minx - focus)
        miny = max(0, miny - focus)
        maxx = min(W, maxx + focus)
        maxy = max(H, maxy + focus)

        # Artiq hemin herf ve reqemleri sekilden ayiririq
        cut = plate_bgr[miny:maxy, minx:maxx].copy()

        # Kesilmis sekli yaddasa yaziriq
        try:
            cv2.imwrite(f"Images/{name}_{index}.jpg", cut)

        except:
            pass

        write = plate_bgr.copy()
        cv2.drawContours(write, [box], 0, (0, 255, 0), 1)

        plt.imshow(write)
        plt.show()
