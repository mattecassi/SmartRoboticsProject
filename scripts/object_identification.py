import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt

FILE_2_ANALYSE = os.path.join("camera_img.jpg")
FILE_2_ANALYSE = os.path.join("/","tmp", "camera_save_tutorial", "img2.jpg")

IMG_BACKGROUND_PATH = "background2.jpg"







class Object_detector():

    def __init__(self, img_background):
        self._bs = cv2.createBackgroundSubtractorKNN()
        self._img_background = img_background
        self._template_green = cv2.imread("green_blocks.jpg",0)
        self._template_red = cv2.imread("red.jpg",0)

        for i in range(20):
            fgMask = self._bs.apply(img_background)

    def _extract_green_points(self, extracted, extracted_gray):
        
     
        w, h = self._template_green.shape[::-1]
        res = cv2.matchTemplate(extracted_gray,self._template_green, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = np.where( res >= threshold)
        matching = zip(*loc[::-1])
        pt2take = []
        for idx, pt in enumerate(matching):
            px2test = pt 
            if extracted[pt[1], pt[0], 1] > 140 and not self._is_redundant_points(pt, pt2take, h, w):
                pt2take.append(pt)
        return np.array(pt2take)

    def _extract_red_points(self, extracted, extracted_gray): 
            w, h = self._template_red.shape[::-1]
            res = cv2.matchTemplate(extracted_gray,self._template_red, cv2.TM_CCOEFF_NORMED)
            threshold = 0.9
            loc = np.where(res >= threshold)
            matching = zip(*loc[::-1])
            pt2take = []
            for idx, pt in enumerate(matching):
                px2test = pt 
                if extracted[px2test[1], px2test[0], -1] > 100 and not self._is_redundant_points(pt, pt2take, h, w):
                    pt2take.append(pt)
                
                # pt2take.append(pt)

            return np.array(pt2take)
        

    def extract_foreground(self, img):
        
        fgMask = self._bs.apply(img)
        # extract position white pixels
        pos_one_mask = np.argwhere(fgMask > 0)
        # cv2.imshow("mask", pos_one_mask.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        max_r, max_c = pos_one_mask.max(axis=0)
        min_r, min_c = pos_one_mask.min(axis=0)
        img_extracted = img[min_r:max_r, min_c :max_c]
        return img_extracted
    
    def _is_redundant_points(self, pt2check, points, H, W):
        pt2check = np.array(pt2check)
        points = np.array(points)
        dist = H
        for pt in points:
            if np.linalg.norm(pt2check - pt) < H:
                return True
        return False


    def template_matching(self, img):
        extracted = self.extract_foreground(img)
        extracted_gray = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
        pt2take_green = self._extract_green_points(extracted, extracted_gray)
        pt2take_red = self._extract_red_points(extracted, extracted_gray)

        for pt in pt2take_green:    
            # print(pt,extracted[pt[0], pt[1]])
            cv2.rectangle(extracted, pt, (pt[0] + self._template_green.shape[0], pt[1] + self._template_green.shape[1]), (0,0,255), 2)

        for pt in pt2take_red:    
            cv2.rectangle(extracted, pt, (pt[0] + self._template_red.shape[0], pt[1] + self._template_red.shape[1]), (0,0,255), 2)

        return pt2take_green, pt2take_red, extracted


class TableCoordinatesConverter():


    def __init__(self):
        self._table_real_len = 0.8
        self._table_coordinates = np.array([0.259017, -0.369258])
    
    def extract_pos_in_table(self, img_extracted_table, pts, inTable=False):
        if pts.shape[0] ==0:
            return None
        m_per_px = self._table_real_len / (img_extracted_table.shape[1])
        add = 0 if inTable else self._table_coordinates
        return pts * m_per_px + add[::-1]








def is_redundant_points(pt2check, points, H, W):
        pt2check = np.array(pt2check)
        points = np.array(points)
        dist = max(H, W)
        for pt in points:
            if np.linalg.norm(pt2check - pt) < dist:
                return True
        return False


def do_template_matching(extracted, template):
    res = cv2.matchTemplate(extracted,template,cv2.TM_CCORR_NORMED)
    threshold = 0.99
    loc = np.where( res >= threshold)
    matching = zip(*loc[::-1])
    pt2take= []
    for pt in matching:
        if not is_redundant_points(pt, pt2take, template.shape[0],template.shape[1 ]):
            pt2take.append(pt)
    return np.array(pt2take) 





def from_image_to_position(img, verbose=False, path_save = None):
    """
    Input ->  Image 
    Output -> Block positions
    """
    
    # 1. Extract foreground and show it
    img_background = cv2.imread(IMG_BACKGROUND_PATH)
    od = Object_detector(img_background)
    foreground = od.extract_foreground(img)
    

    # 2. Extract blocks positions in the image and show them
    green_template = cv2.imread("green_blocks.jpg")
    red_template = cv2.imread("red.jpg")
    pt2take_green = do_template_matching(foreground, green_template)
    pt2take_red = do_template_matching(foreground, red_template)
    w, h = green_template.shape[0], green_template.shape[1] 
    pt2take_green_real = pt2take_green + min(h/2,w/2)
    # show...
    foreground_original = foreground.copy()
    foreground = foreground.copy()
    for pt in pt2take_green:    
        cv2.rectangle(foreground, pt, (pt[0] + h, pt[1] + w), (0,0,255), 2)
    w, h = red_template.shape[0], red_template.shape[1]
    pt2take_red_real = pt2take_red + min(h/2,w/2)

    for pt in pt2take_red:    
        cv2.rectangle(foreground, pt, (pt[0] + h, pt[1] + w), (255,0,0), 2)

    # 3. Convert image positions into 3D positions
    tc = TableCoordinatesConverter()
    pt2take_green_3d = tc.extract_pos_in_table(foreground, pt2take_green_real)
    pt2take_red_3d = tc.extract_pos_in_table(foreground, pt2take_red_real)


    if verbose:
        cv2.imshow("orginal img", img )
        cv2.imshow("foreground", foreground_original)
        cv2.imshow("block tagged", foreground)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if path_save is not None:
        cv2.imwrite(os.path.join(path_save, "orginal_acquisition.jpg"), img )
        cv2.imwrite(os.path.join(path_save, "table_extracted.jpg"), foreground_original)
        cv2.imwrite(os.path.join(path_save, "object_identification.jpg"), foreground)


    return np.round_(pt2take_green_3d,decimals=2), np.round_(pt2take_red_3d, decimals=2)


if __name__ == "__main__":
    img = cv2.imread(FILE_2_ANALYSE)
    pt3d_green, pt3d_red = from_image_to_position(img)
    print(pt3d_green)
    print(pt3d_red)

