import numpy as np
import cv2
import glob

#Extracting the picture for every gap frame of the video

def Video_Slicing(fileName,gap):
    cap = cv2.VideoCapture(fileName)
    i = 0
    while (cap.isOpened()):  # play the video by reading frame by frame
        ret, frame = cap.read()
        if ret == False:
            break

        if i%gap == 0: #Extracting the picture for every gap frames
            cv2.imwrite('calibration' + str(i/10) + '.jpg', frame)
        i = i + 1

def Checkerboard_Calib(fileName,boardHeight,boardWidth):

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((boardHeight*boardWidth,3), np.float32)
    objp[:,:2] = np.mgrid[0:boardWidth, 0:boardHeight].T.reshape(-1,2)

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    # Make a list of calibration images
    images = glob.glob(fileName)

    size = tuple()
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        size = gray.shape[::-1]
    # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (boardWidth,boardHeight), None)
    # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    # Draw and display the corners
            cv2.drawChessboardCorners(img, (boardWidth,boardHeight), corners, ret)
            write_name = 'corners_found'+str(idx)+'.png'
            cv2.imwrite(write_name, img)
            #cv2.imshow('img', img)
            #cv2.imwrite('result1.png',img)
    f1 = open('locations.txt','a')
    f1.write('world coordinates' + str(objpoints) + '\n')
    f1.write('Pixal coordinates' + str(imgpoints) + '\n')

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size,None,None)


    total_error = 0
    for i in range(len(objpoints)):
        points_pixel_repro, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], points_pixel_repro, cv2.NORM_L2) / len(points_pixel_repro)
        total_error += error

    er = total_error / len(objpoints)
    print("Average error of reproject: {}".format(total_error / len(objpoints)))
    return mtx, dist


def undistortion(mtx,dist,fileName):
    img2 = cv2.imread(fileName)
    h, w = img2.shape[:2]
    newcameramtx,roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print(roi)

    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img2, mapx, mapy, cv2.INTER_LINEAR)

    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite('calibresult.png', dst)


#Extracting the nth frame of the video
#@ fileName: the path of Video
#@ num: the number of frame
def Frame_Extraction(fileName,num):
     try:
        vc = cv2.VideoCapture(fileName)
        video_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vc.set(cv2.CAP_PROP_POS_MSEC,num)
        rval,frame = vc.read()
        if rval:
            cv2.imwrite('./Capture.jpg',frame)
        else:
            print('Read Error')

     except Exception as e:
         print('Loading Error')


def pixel_to_world(camera_intrinsics, r, t, img_points):

    K_inv = camera_intrinsics.I
    R_inv = np.asmatrix(r).I

    R_inv_T = np.dot(R_inv, np.asmatrix(t))
    world_points = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_point in img_points:
        coords[0] = img_point[0]
        coords[1] = img_point[1]
        coords[2] = 1.0
        cam_point = np.dot(K_inv, coords)
        cam_R_inv = np.dot(R_inv, cam_point)
        scale = R_inv_T[2][0] / cam_R_inv[2][0]
        scale_world = np.multiply(scale, cam_R_inv)
        world_point = np.asmatrix(scale_world) - np.asmatrix(R_inv_T)
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0] = world_point[0]
        pt[1] = world_point[1]
        pt[2] = 0
        world_points.append(pt.T.tolist())

    return world_points

