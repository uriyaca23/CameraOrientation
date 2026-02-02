import cv2

video_path = r'c:/Users/uriya/PycharmProjects/CameraOrientation/assets/video.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((code >> 8 * i) & 0xFF) for i in range(4)])
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"FourCC_Str: {fourcc_str}")
    print(f"FPS: {fps}")
    print(f"Dimensions: {w}x{h}")
    
cap.release()
