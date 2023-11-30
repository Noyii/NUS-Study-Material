import cv2
import numpy as np

from termcolor import colored
from skimage.metrics import structural_similarity


def extract_color(img, color):
    brightness = 0
    contrast = 10
    brightness += int(round(255*(1-contrast)/2))
    img = cv2.addWeighted(img, contrast, img, 0, brightness)

    img = cv2.GaussianBlur(img, (5, 5), 5)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # set lower and upper color limits
    if color == 'blue':
        lower_val = (90, 50, 20)
        upper_val = (110, 100, 255) # building glass
    elif color == 'grey':
        lower_val = (0, 0, 20)
        upper_val = (180, 15, 255) # grey 
    else:
        print(colored("Colour not specified", "red"))
    
    mask = cv2.inRange(hsv, lower_val, upper_val)
    edges = cv2.Canny(mask, 100, 100)
    # apply mask to original image
    res = cv2.bitwise_and(img, img, mask=mask)

    # detect contours in image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    areas = [0]

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        
        if len(approx) > 4:
            ar = cv2.contourArea(cnt)
            areas.append(ar)
            count+=1
            img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)

    sorted_areas = sorted(areas, reverse=True)
    # diff = sorted_areas[0] - sorted_areas[1]
    return count, sorted_areas[0]
    # return count, (sorted[0] - sorted[1])


def detect_building(img, object='block'):
    count, max_area = extract_color(img, 'grey')
    # count, max_area, diff_area = extract_color(img, 'blue')
    print(max_area)

    if max_area > 30000:
        return True

    # if object == 'glass':
    #     count = extract_color(img, 'grey')
    #     print(count)
    #     if count > 600:
    #         print("Found it!")
    #         return True
    # else:
    #     count = extract_color(img, 'grey')
    #     print(count)
    #     if count > 450:
    #         print("Found it!")
    #         return True
    
    return False


def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count
      
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def preprocess_obs(img):
    image_copy = img.copy() 
    height = img.shape[0]
    width = img.shape[1]

    region_of_interest_vertices = [
        (0, 0),
        (width, 0),
        (width, 2*height/3),
        (width/2, height),
        (0, 2*height/3),
    ]

    roi = region_of_interest(
        image_copy,
        np.array([region_of_interest_vertices], np.int32),
    )

    return roi


def check_image_similarity(first, second):
    # Convert images to grayscale
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    score, diff = structural_similarity(first_gray, second_gray, full=True)
    return score
