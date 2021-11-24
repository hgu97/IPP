import cv2 as cv
import numpy as np
import deburr

#%% img read()
# og_color = cv.imread('D:/Python_code/OG.jpg')    #('D:/Python_code/135m_OG6.jpg')             
# burr_color = cv.imread('D:/Python_code/BURR_Rot&Trans.jpg')#('D:/Python_code/135m_BURR6.jpg')

# og_color = cv.imread('D:/Python_code/test_o.bmp')    #('D:/Python_code/135m_OG6.jpg')     
# burr_color = cv.imread('D:/Python_code/test_b.bmp')    #('D:/Python_code/135m_OG6.jpg')     

og_color = cv.imread('C:/Users/GL-NT/opencv/test_o.bmp') 
burr_color = cv.imread('C:/Users/GL-NT/opencv/test.bmp') 


og = cv.cvtColor(og_color, cv.COLOR_BGR2GRAY)
burr = cv.cvtColor(burr_color, cv.COLOR_BGR2GRAY)


#%% 이미지 trim
og_upper, og_lower, og_left, og_right = deburr.imtrim(og,n=1,offset=100)

og_trim = og_color[og_upper:og_lower, og_left:og_right]
og_trim_g = og[og_upper:og_lower, og_left:og_right]
trim_og = og_trim.copy()
test_og= og_trim.copy()
       
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
o = test_bg.copy()
b = test_bg.copy()

# cv.drawContours(og_trim, og_largestcnt, -1, (255,0,0), 1)    
# cv.drawContours(og_trim, rot_largestcnt, -1, (0,0,255), 1)  
    
cv.drawContours(test_bg, og_largestcnt, -1, (255,0,0), 1)    
cv.drawContours(test_bg, rot_largestcnt, -1, (0,255,255), 1)  
cv.drawContours(o, og_largestcnt, -1, (255,0,0), 1)    
cv.drawContours(b, rot_largestcnt, -1, (0,255,255), 1)  
   
cv.drawContours(trim_og, og_largestcnt, -1, (0,255,0), 1)

#%% burr size
tool_feed = deburr.burr_size(og_largestcnt,rot_largestcnt, test_bg)
  
# i =0
# test_og[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][0]=255 ## B
# test_og[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][1]=0   ## G 
# test_og[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][2]=0   ## R 

cv.circle(test_og,(tuple(og_largestcnt[0].ravel())),2,(0,255,0),-1, cv.LINE_AA) #red

cv.circle(test_og,(tuple(og_largestcnt[771].ravel())),2,(0,255,0),-1, cv.LINE_AA) #red

cv.circle(test_og,(tuple(og_largestcnt[500].ravel())),2,(0,0,255),-1, cv.LINE_AA) #red

cv.circle(test_og,(tuple(og_largestcnt[920].ravel())),2,(0,0,255),-1, cv.LINE_AA) #red

cv.circle(test_og,tuple([291,334]),3,(255,0,0),-1, cv.LINE_AA) #red

# cv.imshow('trim_og',trim_og)
# cv.imshow('test_bg',test_bg)
# cv.imshow('og_trim',og_trim)
# cv.imshow('burr_trim',burr_trim)
# cv.imshow('o',o)
# cv.imshow('b',b)
cv.imshow('test_og',test_og)

cv.waitKey()
cv.destroyAllWindows()