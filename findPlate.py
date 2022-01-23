import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

car_image = os.listdir("Cars")
img = cv2.imread("Cars/" + car_image[0])
img = cv2.resize(img, (500, 500))


def findCarPlate(img):
    # Çoxlu rənglərdən xilas olmaq üçün
    img_bgr = img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Lazımsız qırışları aradan qaldırıriq
    # process image = p_img
    p_img = cv2.medianBlur(img_gray, 5)
    p_img = cv2.medianBlur(p_img, 5)

    # Şəklin medianini tapırıq
    median = np.median(p_img)

    low = 0.67 * median  # 3/2
    high = 1.33 * median  # 4/3

    # Kenarlari tapiriq
    edge = cv2.Canny(p_img, low, high)

    # Sekli genislendiririk / to expand the shape
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)

    # Sekilde ardicil gelen 255'leri yeni ag rengleri tapiriq
    # RETR_TREE -> iyerarxiya quruluşu
    # CHAIN_APPROX_SIMPLE -> yalniz kenarli olanlar
    cnt = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnt[0]
    cnt = sorted(cnt, key=cv2.contourArea, reverse=True)

    H, W = 500, 500

    carPlate = None

    # Sekildeki kvadratlari tapib icerisinden nomreni secirik
    for c in cnt:
        rect = cv2.minAreaRect(c)
        (x, y), (w, h), r = rect
        if (w > h and w > h * 2) or (h > w * 2 and h > w * 2):
            # Tepe noqtelerini tapiriq
            box = cv2.boxPoints(rect)
            box = np.int64(box)

            minx = np.min(box[:, 0])
            miny = np.min(box[:, 1])
            maxx = np.max(box[:, 0])
            maxy = np.max(box[:, 1])

            # Nomre ola bilecek ehtimali tapiriq
            probPlate = img_gray[miny:maxy, minx:maxx].copy()
            probMedian = np.median(probPlate)

            control1 = probMedian > 85 and probMedian < 200
            control2 = h < 50 and w < 150
            control3 = w < 150 and h < 150

            print(f"Nomrenin texmini mediani: {probMedian}, genislik: {w}, yukseklik: {h}")
            # Bu sertin icine girirse demeli nomreni tapibdi alqortim
            control = False
            if (control1 and (control2 or control3)):
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                # Burada nomrenin tepe noqtelerini qeyd edirik
                plate = [int(i) for i in [minx, miny, w, h]]
                control = True
                print("Nomre tapildi")

            else:
                cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
                print("Nomre tapilmadi")

            if control:
                return plate
                break
    return []


findCarPlate(img)
