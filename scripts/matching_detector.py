import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


class MatchingDetector:
    def __init__(self,mode) -> None:
        self.mode = mode
        
    def template_matching_image(self,template_path, image_path, method=cv.TM_CCOEFF_NORMED):
        img = cv.imread(image_path,cv.COLOR_BGR2GRAY)
        
        template = cv.imread(template_path,cv.COLOR_BGR2GRAY)
        w, h = template.shape[0], template.shape[1]
        
        if w > img.shape[0] or h > img.shape[1]:
            return None, None, None, None,None
        # All the 6 methods for comparison in a list
       
        # method = eval(method)
        # Apply template Matching
        res = cv.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res) 
        
        location = (w,h)+min_loc+max_loc
        
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
       
        return location, bottom_right,top_left, res, img
    
    def get_location(self,res):
       min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
       return min_loc, max_loc 
    
    def plot_matches(self,res,img, location, method=cv.TM_CCOEFF_NORMED):
        w, h, min_loc, max_loc = location[0],location[1],(location[2],location[3]),(location[4],location[5])
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        cv.rectangle(img, top_left, bottom_right, 255, 4)
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(method)
        plt.show()
        