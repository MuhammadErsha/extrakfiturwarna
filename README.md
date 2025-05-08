# extrakfiturwarna
import cv2
import numpy as np

def detect_shapes(image_path):
    # Baca gambar
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Cari kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Approximate the shape
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        x, y, w, h = cv2.boundingRect(approx)
        cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)

        shape = "Unknown"
        sides = len(approx)

        if sides == 3:
            shape = "Segitiga"
        elif sides == 4:
            # Cek apakah persegi/kotak atau trapesium
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape = "Persegi"
            else:
                shape = "Trapesium atau Kotak"
        elif sides > 4:
            # Gunakan deteksi lingkaran
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > 0.8:
                shape = "Lingkaran"
            else:
                shape = "Poligon"

        # Tampilkan label bentuk
        cv2.putText(img, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Tampilkan hasil
    cv2.imshow("Shapes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ganti dengan path gambar kamu
detect_shapes("shapes.png")
