import cv2
import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim
import numpy as np
import pywhatkit as kit

kit.image_to_ascii_art(r'C:\Users\82105\Desktop\opencv\2.jpg', r"ascii art2")
kit.image_to_ascii_art(r'C:\Users\82105\Desktop\opencv\3.jpg', r"ascii art3")
kit.image_to_ascii_art(r'C:\Users\82105\Desktop\opencv\4.jpg', r"ascii art4")

def ascii_to_image(ascii_art):
    lines = ascii_art.split('\n')

    height = len(lines)
    width = max(len(line) for line in lines)
    

    img = np.ones((height * 10, width * 10, 3), dtype=np.uint8) * 255 
    
   
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char != ' ':  
                cv2.putText(img, char, (x * 10, y * 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return img

def similarity_between_ascii(ascii_art1, ascii_art2):

    img1 = ascii_to_image(ascii_art1)
    img2 = ascii_to_image(ascii_art2)
    

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    

    sim, _ = ssim(gray1, gray2, full=True)
    
    return sim


ascii_art2 = "ascii art1.txt"


ascii_art3 = "ascii art3.txt"

ascii_art4 = "ascii art4.txt"


# Calculate similarity
similarity = similarity_between_ascii(ascii_art2, ascii_art3)
print(f"2와 3아스키 아트 사이의 유사성: {similarity*100}%")

similarity = similarity_between_ascii(ascii_art3, ascii_art4)
print(f"3과 4아스키 아트 사이의 유사성: {similarity*100}%")
