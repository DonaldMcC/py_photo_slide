import sys, cv2, numpy as np

def order_pts(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def four_point_warp(image, pts):
    rect = order_pts(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH), flags=cv2.INTER_CUBIC)
    return warped


def find_screen_quad(img):
    h, w = img.shape[:2]
    scale = 1000.0 / h
    small = cv2.resize(img, (int(w*scale), 1000))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2).astype("float32")
            pts /= scale

    return pts

#return None

def enhance(img):
    # White balance (try xphoto gray-world if available)
    try:
        wb = cv2.xphoto.createSimpleWB()
        wb.setP(0.5)
        img = wb.balanceWhite(img)
    except Exception:
        pass

    # Contrast/clarity
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab = cv2.merge([l2, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Mild denoise + sharpen
    img = cv2.bilateralFilter(img, 7, 50, 50)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)
    return img


def trim_safe(img, pct=0.02):
    h, w = img.shape[:2]
    dy, dx = int(hpct), int(wpct)
    return img[dy:h-dy, dx:w-dx]


def main(inp, outp):
    img = cv2.imread(inp)
    if img is None:
        raise SystemExit("Could not read input image.")
    quad = find_screen_quad(img)

    if quad is None:
        # Fallback: gentle rectification by assuming near-rectangular central screen
        print("Auto-detect failed; using centered crop fallback.")
        h, w = img.shape[:2]
        m = int(min(h, w)*0.05)
        crop = img[m:h-m, m:w-m]
        warped = crop
    else:
        warped = four_point_warp(img, quad)
        # Slight inner crop to remove any thin black edges
        warped = trim_safe(warped, 0.01)

        # If portrait by mistake, rotate
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    cleaned = enhance(warped)
    cv2.imwrite(outp, cleaned)
    print(f"Saved: {outp}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_slide.py input.jpg output.png")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
