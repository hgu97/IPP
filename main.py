import cv2 as cv
import numpy as np
import deburr
from robodk import *
from robolink import *

#%% img read()
# og_color = cv.imread('D:/Python_code/OG.jpg')    #('D:/Python_code/135m_OG6.jpg')             
# burr_color = cv.imread('D:/Python_code/BURR_Rot&Trans.jpg')#('D:/Python_code/135m_BURR6.jpg')

# og_color = cv.imread('D:/Python_code/test_o.bmp')    #('D:/Python_code/135m_OG6.jpg')     
# burr_color = cv.imread('D:/Python_code/test_b.bmp')    #('D:/Python_code/135m_OG6.jpg')     

og_color = cv.imread('C:/Users/GL-NT/opencv/test_o.bmp') 
burr_color = cv.imread('C:/Users/GL-NT/opencv/test.bmp') 


#%% img resize
og_width = 800         
og_height = 600 
og_color = cv.resize(og_color, (og_width,og_height),interpolation = cv.INTER_AREA)
burr_color = cv.resize(burr_color, (og_width,og_height), interpolation = cv.INTER_AREA)
og = cv.cvtColor(og_color, cv.COLOR_BGR2GRAY)
burr = cv.cvtColor(burr_color, cv.COLOR_BGR2GRAY)


#%% 이미지 trim
og_upper, og_lower, og_left, og_right = deburr.imtrim(og,n=1,offset=40)

og_trim = og_color[og_upper:og_lower, og_left:og_right]
og_trim_g = og[og_upper:og_lower, og_left:og_right]
trim_og = og_trim.copy()


#%% Template matching
cv.imshow('a',og_trim)
burr_left, burr_upper, burr_right, burr_lower = deburr.TempMatch(og_trim_g,burr)

burr_trim = burr_color[burr_upper:burr_lower,burr_left:burr_right]
burr_trim_g = burr[burr_upper:burr_lower,burr_left:burr_right]
trim_burr = burr_trim.copy()


#%% Alinge을 위한 match함수  """ Feature match """
matches,matchesMask, kp1, kp2,src,dst = deburr.F_match(og_trim_g,burr_trim_g)
res2 = cv.drawMatches(og_trim, kp1, burr_trim, kp2, matches, None, \
                    matchesMask = matchesMask,
                    flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


#%% Alinge을 위한 knn match함수
good_matches, knn_matchesMask, kp1, kp2, knn_src, knn_dst = deburr.F_knnmatch(og_trim_g,burr_trim_g)
res4 = cv.drawMatches(og_trim, kp1, burr_trim, kp2, good_matches, None, \
                    matchesMask = knn_matchesMask,
                    flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
  
    
#%%   Align
"""###############  (knn, match 수정)  ######################"""
M_theta1 = deburr.Align(src,dst,matchesMask)
cv.imshow('Matching-Homo', res2)

# M_theta1 = deburr.Align(knn_src,knn_dst,knn_matchesMask)
# cv.imshow('Matching-knn_match-Homo', res4)
rows,cols = burr_trim.shape[0:2]
img_rot = cv.warpAffine(burr_trim, M_theta1,(cols, rows), \
                        flags=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT)
  
    
#%%  Contour drawing
img_rot_g = cv.cvtColor(img_rot, cv.COLOR_BGR2GRAY)

rot_largestcnt,_ = deburr.contour(img_rot_g)
og_largestcnt,_ = deburr.contour(og_trim_g)
        
test_bg = np.zeros(og_trim.shape,dtype=np.uint8) 

# cv.drawContours(og_trim, og_largestcnt, -1, (255,0,0), 1)    
# cv.drawContours(og_trim, rot_largestcnt, -1, (0,0,255), 1)  
    
cv.drawContours(test_bg, og_largestcnt, -1, (255,0,0), 1)    
cv.drawContours(test_bg, rot_largestcnt, -1, (0,0,255), 1)  
   
cv.drawContours(trim_og, rot_largestcnt, -1, (0,255,0), 1)

#%% burr size
tool_feed = deburr.burr_size(og_largestcnt,rot_largestcnt, trim_og)
  

# 결과 출력                    
cv.imshow('test_bg', test_bg)
cv.imshow('trim_og',trim_og)
cv.imshow('og_color', og_color)
cv.imshow('burr_color', burr_color)

cv.waitKey()
cv.destroyAllWindows()

# Start the RoboDK API:
RDK = Robolink()

# Get the robot item by name:
robot = RDK.Item('Fanuc LR Mate 200iD', ITEM_TYPE_ROBOT)

# Get the reference target by name:
target = RDK.Item('Target')
Home = RDK.Item('Target1')
target_pose = target.Pose()
xyz_ref = target_pose.Pos()

robot.setSpeed(10,5,-1,-1)
robot.MoveJ(Home)

# Move the robot to the reference point:
robot.MoveJ(target)

# Draw a hexagon around the reference target:
for i in range(len(og_largestcnt)):
    if tool_feed[i]>=5:
        robot.setSpeed(1,-1,-1,-1)
    elif tool_feed[i]>=4:
        robot.setSpeed(1,-1,-1,-1)
    elif tool_feed[i]>=3:
        robot.setSpeed(1,-1,-1,-1)
    elif tool_feed[i]>=2:
        robot.setSpeed(500,-1,-1,-1)
    elif tool_feed[i]>=1:
        robot.setSpeed(500,-1,-1,-1)
        
    # Calculate the new position around the reference:
    x = int(og_largestcnt[i,:,0])+xyz_ref[0]  # new X coordinate
    y = int(og_largestcnt[i,:,1])+xyz_ref[1]  # new Y coordinate
    z = xyz_ref[2]              # new Z coordinate
    target_pose.setPos([x,y,z])

    print(x, y, z)

    # Move to the new target:
    robot.MoveL(target_pose)

# Trigger a program call at the end of the movement

# Move back to the reference target:
robot.setSpeed(10,5,-1,-1)
robot.MoveL(target)
robot.MoveJ(Home)