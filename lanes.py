from operator import le
import cv2
from cv2 import Canny
import numpy as np
import matplotlib.pyplot as plt




def canny(image):
                                                            #CANNY EDGE DETECTION( identify sharp changes in intensity in the adjacebt pixels)   

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)     
    
    blur=cv2.GaussianBlur(gray, (5,5),0)                           #it apply the gaussian blur on image in 5x5 kernel, and leaves the deviation 0 
    
    canny= cv2.Canny(blur, 50,150)                                 #it will detect the strongest gratient difference in the image and then outline the edges with white line 

    return canny


def region_of_interest(image):

    height = image.shape[0]
    triangle = np.array([[(200,height), (1100,height), (550, 250)]])   #verteces of tringle of our region of interest 
    mask = np.zeros_like(image)                                      #create an array of zeros with the same shape as the image corresponding array   
    cv2.fillPoly(mask, triangle, 255)                                #apply the triangle on the mask  such that area bounded with the polygon will be completely white
    
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):                                        
    line_image= np.zeros_like(image)                                      
    if lines is not None:
        for x1,y1,x2,y2  in lines:
            cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0), 5)
    return line_image


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope< 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    print(left_fit_average, 'left')
    print(right_fit_average, 'right')
    left_line =  make_coordinates(image, left_fit_average)
    right_line =  make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])




def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1= image.shape[0]
    y2= int(y1*(3/5))
    x1= int((y1 - intercept)/slope)
    x2= int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])





# image = cv2.imread('test_image.jpg')                                #read the image from our file and and return as multidimensional numpy array containing relative intensity of each pixel in the image
                                                
# lane_image= np.copy(image)                                          #we made copy of the real image because in the future if we make any change in image it will reflect on the lane image which we dont want

# canny = canny(lane_image)
# cropped_image = region_of_interest(canny)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]),minLineLength=10 , maxLineGap=10 )    #HOUGH TRANSFORMATION
# averaged_lines= average_slope_intercept(lane_image, lines)
# line_image= display_lines(lane_image, averaged_lines)

# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

# cv2.imshow('result', combo_image)
# #plt.imshow(canny)
# cv2.waitKey(0)
# #plt.show()

cap= cv2.VideoCapture("sample vid.mp4")
while(cap.isOpened()):
    _, frame=cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]),minLineLength=40 , maxLineGap=5 )    #HOUGH TRANSFORMATION
    averaged_lines= average_slope_intercept(frame, lines)
    line_image= display_lines(frame, averaged_lines)

    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow('result', combo_image)
    #plt.imshow(canny)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()





