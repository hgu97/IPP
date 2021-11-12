import cv2 as cv
import numpy as np
import math

#%% largest contour
def contour(img):
    _, temp = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) 
    thr = cv.adaptiveThreshold(temp,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,3,2)
    thr = cv.bitwise_not(thr)
    contours,_ = cv.findContours(thr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)        
    largestcnt = []    
    for cnt in contours:
        if (len(cnt) > len(largestcnt)):
            largestcnt = cnt  
 
    return largestcnt, contours

#%% match() or knnmatch()
def Align(src_pts,dst_pts,matchesMask):
    pos = [i for i in range(len(matchesMask)) if matchesMask[i]==1]   
    src = src_pts[pos]
    dst = dst_pts[pos]   
    dx = ( dst.mean(axis=0)[0] - src.mean(axis=0)[0] )
    dy = ( dst.mean(axis=0)[1] - src.mean(axis=0)[1] )

    vec_s = src - src.mean(axis=0)
    vec_d = dst - dst.mean(axis=0)            
    norm_s = vec_s * vec_s
    norm_s = np.sqrt(norm_s.sum(axis=1))
    norm_d = vec_d * vec_d
    norm_d = np.sqrt(norm_d.sum(axis=1))
    
    inner=[]
    for i in range(len(vec_s)):
        inner.append(vec_s[i][0]*vec_d[i][1] - vec_s[i][1]*vec_d[i][0])
    v = np.array(inner) / (norm_s * norm_d)
    theta = np.arcsin(v)
    theta = theta.mean() *180/math.pi
    
    Matrix = cv.getRotationMatrix2D((dst.mean(axis=0)[0],dst.mean(axis=0)[1]),theta,1)
    Matrix[0][2] = Matrix[0][2] -dx
    Matrix[1][2] = Matrix[1][2] -dy
    return Matrix

#%% img trim 
def imtrim(img,n,offset):
    dst = np.zeros(img.shape, np.uint8)
    # 가로 세로 n등분 하기
    bw = img.shape[1] // n
    bh = img.shape[0] // n
    
    for y in range(n):
        for x in range(n):
            img_ = img[y*bh:(y+1)*bh, x*bw:(x+1)*bw]   # threshold 입력값으로 주기 위해 입력 영상도 등분
            dst_ = dst[y*bh:(y+1)*bh, x*bw:(x+1)*bw] # dst_를 변경하면 dst도 변경됍니다.
            
            cv.threshold(img_, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU, dst_)
    thr = cv.adaptiveThreshold(dst,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,3,2)
    thr = cv.bitwise_not(thr)
    
    contours,_ = cv.findContours(thr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)        
    largestcnt = []    
    for cnt in contours:
        if (len(cnt) > len(largestcnt)):
            largestcnt = cnt     
    
    contours_xy = np.array(largestcnt)
    # x,y의 min과 max 찾기
    x_min, x_max, y_min, y_max = 0,0,0,0
    value = list()
    for i in range(len(contours_xy)):
            value.append(contours_xy[i][0][0]) #세번째 괄호가 0일때 x의 값
            x_min = min(value)
            x_max = max(value)
    value = list()
    for i in range(len(contours_xy)):
            value.append(contours_xy[i][0][1]) #세번째 괄호가 1일때 y의 값
            y_min = min(value)
            y_max = max(value)
    
    upper =  y_min-offset
    lower = y_max+offset
    left = x_min-offset
    right = x_max+offset
    
    return upper, lower, left, right

#%% Template Matching
def TempMatch(img1,img2):
    
    th, tw = img1.shape[:2]
    
    methods = ['cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR_NORMED', \
                                         'cv.TM_SQDIFF_NORMED']
    for i, method_name in enumerate(methods):
        img_draw = img2
        method = eval(method_name)
        # 템플릿 매칭   ---①
        res = cv.matchTemplate(img2, img1, method)
        # 최대, 최소값과 그 좌표 구하기 ---②
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # TM_SQDIFF의 경우 최소값이 좋은 매칭, 나머지는 그 반대 ---③
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
            match_val = min_val
        else:
            top_left = max_loc
            match_val = max_val
        # 매칭 좌표 구해서 사각형 표시   ---④      
        bottom_right = (top_left[0] + tw, top_left[1] + th)
        cv.rectangle(img_draw, top_left, bottom_right, (255,255,255),2)
        # # 매칭 포인트 표시 ---⑤
        cv.putText(img_draw, str(match_val), top_left, \
                    cv.FONT_HERSHEY_PLAIN, 2,(255,255,0), 1, cv.LINE_AA)
        cv.circle(img_draw,max_loc,10,(255,255,255),-1, cv.LINE_AA) # matching point
       
        
    (left, upper) = top_left
    (right, lower) = bottom_right
    return left, upper, right, lower


#%% Feature Match()
def F_match(gray1,gray2):
    
    # ORB로 서술자 추출 ---①
    detector = cv.ORB_create()
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    # BF-Hamming으로 매칭 ---②
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    
    # 매칭 결과를 거리기준 오름차순으로 정렬 ---③
    matches = sorted(matches, key=lambda x:x.distance)
    
    # 매칭점으로 원근 변환 및 영역 표시 ---⑤
    src = np.float32([ kp1[m.queryIdx].pt for m in matches ])
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches ])
    
    # RANSAC으로 변환 행렬 근사 계산 ---⑥
    mtrx, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
    h,w = gray1.shape
    # pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
    # dst1 = cv.perspectiveTransform(pts,mtrx)
    
    # 정상치 매칭만 그리기 ---⑦
    matchesMask = mask.ravel().tolist()
    
    return matches,matchesMask, kp1, kp2, src, dst

#%% Feature knnMatch()
def F_knnmatch(gray1,gray2):
    
    # ORB로 서술자 추출 ---①
    detector = cv.ORB_create()
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    # BF-Hamming 생성 ---②
    matcher = cv.BFMatcher(cv.NORM_HAMMING2)
    # knnMatch, k=2 ---③
    matches = matcher.knnMatch(desc1, desc2, 2)
    
    # 첫번재 이웃의 거리가 두 번째 이웃 거리의 70% 이내인 것만 추출---⑤     ######################## 퍼센티지 수정가능
    ratio1 = 0.6
    ratio2 = 0.8
    good_matches = [first for first,second in matches \
                        if first.distance < second.distance * ratio2 and first.distance  > second.distance* ratio1]
        
    # 좋은 매칭점의 queryIdx로 원본 영상의 좌표 구하기 ---6
    knn_src = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
    # 좋은 매칭점의 trainIdx로 대상 영상의 좌표 구하기 ---7
    knn_dst = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
    
    # RANSAC으로 변환 행렬 근사 계산 ---⑥
    mtrx, mask = cv.findHomography(knn_src, knn_dst, cv.RANSAC, 5.0)
    # 정상치 매칭만 그리기 ---⑦
    knn_matchesMask = mask.ravel().tolist()
    
    return good_matches, knn_matchesMask, kp1, kp2, knn_src, knn_dst

#%% burr size
def burr_size(og_largestcnt, rot_largestcnt, trim_og):

    og_con_xy = og_largestcnt[:,0,:]
    og_con_x = og_largestcnt[:,0,:][:,0]
    og_con_y = og_largestcnt[:,0,:][:,1]
    
    burr_con_xy = rot_largestcnt[:,0,:]
    burr_con_x = rot_largestcnt[:,0,:][:,0]
    burr_con_y = rot_largestcnt[:,0,:][:,1]
    
    dis_min = []
    tool_feed=[]
    burr_size=2
    
    """    Burr size idea1(상, 하, 좌, 우 길이 비교)   """
    for i in list(range(0,len(og_largestcnt[:,:,0]))):
        y_dis_min = []
        ind_y = np.where(burr_con_x.ravel() == og_con_x[i])
        aim_y = burr_con_y[ind_y]
        x_dis_min = []
        ind_x = np.where(burr_con_y.ravel() == og_con_y[i])
        aim_x = burr_con_x[ind_x]
        dis_min.append(min(np.append(abs(aim_y-og_con_y[i]),abs(aim_x-og_con_x[i]))))
        
        
        ## 원으로 디버링이 필요한 점 찍기, 가시성 증대
        if dis_min[i]>=5:
            cv.circle(trim_og,(tuple(og_largestcnt[i].ravel())),1,(0,0,255),-1, cv.LINE_AA) #red
        elif dis_min[i]>=4:
            cv.circle(trim_og,(tuple(og_largestcnt[i].ravel())),1,(0,165,255),-1, cv.LINE_AA) # orange
        elif dis_min[i]>=3:
            cv.circle(trim_og,(tuple(og_largestcnt[i].ravel())),1,(0,255,255),-1, cv.LINE_AA) # yellow
        elif dis_min[i]>=2:
            cv.circle(trim_og,(tuple(og_largestcnt[i].ravel())),1,(0,128,0),-1, cv.LINE_AA) # green
        elif dis_min[i]>=1:
            cv.circle(trim_og,(tuple(og_largestcnt[i].ravel())),1,(255,0,0),-1, cv.LINE_AA) # blue
            
        # ### 점으로 디버링이 필요한 점 찍기, 정확도 증대
        # if dis_min[i]>=5:
        #     og_trim[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][0]=0   ## B
        #     og_trim[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][1]=0   ## G 
        #     og_trim[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][2]=255 ## R 
        # elif dis_min[i]>=4:
        #     og_trim[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][0]=0   ## B
        #     og_trim[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][1]=165 ## G 
        #     og_trim[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][2]=255 ## R 
        # elif dis_min[i]>=3:
        #     og_trim[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][0]=0   ## B
        #     og_trim[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][1]=255 ## G 
        #     og_trim[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][2]=255 ## R 
        # elif dis_min[i]>=2:
        #     og_trim[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][0]=0   ## B
        #     og_trim[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][1]=128 ## G 
        #     og_trim[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][2]=0   ## R 
        # elif dis_min[i]>=1:
        #     og_trim[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][0]=255 ## B
        #     og_trim[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][1]=0   ## G 
        #     og_trim[og_largestcnt[i,0,1]][og_largestcnt[i,0,0]][2]=0   ## R 
              
    return dis_min