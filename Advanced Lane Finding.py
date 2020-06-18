#!/usr/bin/env python
# coding: utf-8

# ## Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. (DONE)
# * Apply a distortion correction to raw images. (DONE)
# * Use color transforms, gradients, etc., to create a thresholded binary image. (DONE)
# * Apply a perspective transform to rectify binary image ("birds-eye view"). (DONE)
# * Detect lane pixels and fit to find the lane boundary. (JUST A BIT TWEAK)
# * Determine the curvature of the lane and vehicle position with respect to center. (DONE)
# * Warp the detected lane boundaries back onto the original image. (DONE)
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position. (IN PROGRESS)
# 
# ---

# ## (Not important) Tune the parameters for threshold

# In[ ]:


from IPython.html import widgets
from IPython.html.widgets import interact
from IPython.display import display
image = mpimg.imread('../test_images/test1.jpg')
image = cv2.undistort(image, mtx, dist, None, mtx)

def interactive_mask(s_low, s_high, sx_low, sx_high, b_low, b_high, l_low, l_high):
    #combined = combined_binary_mask(image, ksize, mag_low, mag_high, dir_low, dir_high,\
                                    #hls_low, hls_high, bright_low, bright_high)
    combined = threshold(image, s_thresh=(0,255), sx_thresh=(0,255), b_thresh=(0,255), l_thresh=(0,255))
    plt.figure(figsize=(10,10))
    plt.imshow(combined,cmap='gray')
interact(interactive_mask, s_low=(0,255), s_high=(0,255),         sx_low=(0, 255), sx_high=(0, 255), b_low=(0,255),         b_high=(0,255), l_low=(0,255), l_high=(0,255))


# ## Compute the camera calibration matrix and distortion coefficients given a set of chessboard images

# In[1]:


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = mpimg.imread(fname)
    #img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(mpimg.imread(fname))
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(img)
    ax2.set_title('Chessboard Corners Image', fontsize=20)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[1], img.shape[0]), None, None)


# ## Apply a distortion correction to raw images

# In[2]:


def cal_undistort(img):
    # Use cv2.calibrateCamera() and cv2.undistort()
    #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[1], img.shape[0]), None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

raw_images = glob.glob('../test_images/test*.jpg')
for fname in raw_images:
    img = mpimg.imread(fname)
    undist = cal_undistort(img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    f.tight_layout()
    ax1.imshow(mpimg.imread(fname))
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# ## Use Color and Gradient Thresholds to create threshold binary images

# In[4]:


def threshold(undist, sobel_kernel=5, s_thresh=(230, 255), mag_thresh=(230, 255), b_thresh=(150,200), l_thresh=(230,255)):
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
    
    # Sobel operators
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    lab = cv2.cvtColor(undist, cv2.COLOR_RGB2LAB)
    b_channel = lab[:,:,2]
    
    luv = cv2.cvtColor(undist, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    
    # Threshold x gradient
    mag_output = np.zeros_like(gradmag)
    mag_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    
    # Threshold hls color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Threshold lab color channel
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1
    
    # Threshold luv color channel
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(mag_output)
    combined_binary[(s_binary == 1) | (mag_output == 1) | (b_binary == 1) | (l_binary == 1)] = 1
    return combined_binary


threshold_images = glob.glob('../test_images/test*.jpg')
for fname in threshold_images:
    image = mpimg.imread(fname)
    undist = cal_undistort(image)
    #hls_binary = hls_select(image, thresh=(0, 150))
    thresholded_image = threshold(undist)
    
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(undist)
    ax1.set_title('Undistorted Image', fontsize=50)
    ax2.imshow(thresholded_image, cmap='gray')
    ax2.set_title('Thresholded Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    


# ## Apply Perspective Transform (warp) to rectify binary images

# In[5]:


def perspective_transform(binary_img):
    # Define calibration box in source (original) and destination (desired or warped) coordinates
    img_size = (binary_img.shape[1],binary_img.shape[0])
    #'''
    src = np.float32([(0,680),
                      (1280,680), 
                      (800,480), 
                      (550,480)])
    dst = np.float32([(0,720),
                      (1280,720),
                      (1280,0),
                      (0,0)])
    #'''
    '''
    # Four source coordinates
    src = np.float32(
        [[100,720],
         [1200,720],
         [800,500],
         [500,500]])
    
    # Four desired coordinates
    dst = np.float32(
        [[200,720],
         [1000,720],
         [1000,0],
         [200,0]])
         '''
    
    # Compute the perspective transform, M (the transformation matrix)
    M = cv2.getPerspectiveTransform(src,dst)
    
    # Create warped image - uses linear interpolation
    warped = cv2.warpPerspective(binary_img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped

'''
detect_image4 = mpimg.imread('../test_images/test2.jpg')
undist4 = cal_undistort(detect_image4)
threshold4 = threshold(undist4)
warped4 = perspective_transform(threshold4)
plt.imshow(warped4,  cmap='gray')
'''

unwarped_images = glob.glob('../test_images/test*.jpg')
for fname in unwarped_images:
    image = mpimg.imread(fname)
    undist = cal_undistort(image)
    thresholded_image = threshold(undist)
    warped_img = perspective_transform(thresholded_image)
    
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(undist)
    ax1.set_title('Undistorted Image', fontsize=50)
    ax2.imshow(warped_img, cmap='gray')
    ax2.set_title('Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# ## Detect lane pixels and fit to find the lane boundary

# In[6]:


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 10
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def visual_fit_poly(out_img, left_fitx, right_fitx, ploty):
    ## Visualization ##
    
    # Plots the left and right polynomials on the lane lines
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, out_img.shape[1])
    plt.ylim(out_img.shape[0],0)
    return None

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit` ###
    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    #plt.imshow(out_img)
    
    #visual_fit_poly(out_img, left_fitx, right_fitx, ploty)
    '''
    # Plots the left and right polynomials on the lane lines
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, binary_warped.shape[1])
    plt.ylim(binary_warped.shape[0],0)
    '''

    return out_img, left_fitx, right_fitx, ploty

'''
detect_image1 = mpimg.imread('../test_images/test1.jpg')
undist5 = cal_undistort(detect_image1)
threshold5 = threshold(undist5)
warped5 = perspective_transform(threshold5)
out_img, left_fitx, right_fitx, ploty = fit_polynomial(warped5)
visual_fit_poly(out_img, left_fitx, right_fitx, ploty)
'''

detect_images = glob.glob('../test_images/test*.jpg') 
for fname in detect_images:
    img = mpimg.imread(fname)
    undist = cal_undistort(img)
    thresholded_image = threshold(undist)
    warped_img = perspective_transform(thresholded_image)
    fit_image, place_holder1, place_holder2, place_holder3 = fit_polynomial(warped_img)
    f, (ax1, ax2) = plt.subplots(1,2,figsize=(20,10))
    ax1.set_title("Source image")
    ax1.imshow(cal_undistort(img))
    ax2.set_title("Fit image")
    ax2.imshow(fit_image)


# ## Search around poly

# In[7]:


def search_around_poly(binary_warped):
    leftx, lefty, rightx, righty, place_holder = find_lane_pixels(binary_warped)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    # Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 75

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    ## Visualization
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, binary_warped.shape[1])
    plt.ylim(binary_warped.shape[0],0)
    
    return result

test_image2 = mpimg.imread('../test_images/test2.jpg')
undist6 = cal_undistort(test_image2)
threshold6 = threshold(undist6)
warped6 = perspective_transform(threshold6)
out_img = search_around_poly(warped6)


# ## Unwarp image

# In[8]:


def unwarp(warped):
    src = np.float32([(0,680),
                      (1280,680), 
                      (800,480), 
                      (550,480)])
    dst = np.float32([(0,720),
                      (1280,720),
                      (1280,0),
                      (0,0)])
    
    '''
    src = np.float32(
        [[100,720],
         [1200,720],
         [800,500],
         [500,500]])
    
    # Four desired coordinates
    dst = np.float32(
        [[200,720],
         [1000,720],
         [1000,0],
         [200,0]])    
    '''
    Minv = cv2.getPerspectiveTransform(dst,src)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    unwarped = cv2.warpPerspective(warped, Minv, (img.shape[1], img.shape[0]))
    return unwarped, Minv


# ## Project lines onto original images

# In[9]:


def projection(img):
    undist = cal_undistort(img)
    thresholded = threshold(undist)
    warped = perspective_transform(thresholded)
    out_img, left_fitx, right_fitx, ploty = fit_polynomial(warped)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))*255

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp, Minv = unwarp(color_warp)
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    return None

test_image = mpimg.imread('../test_images/test6.jpg')
projection(test_image)


# ## Measure the real curvatuve

# In[10]:


def measure_curvature_real(binary_warped):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Get the arrays of left and right lane lines
    leftx, lefty, rightx, righty, place_holder = find_lane_pixels(binary_warped)
    
    # Define fit polynomial in meters
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Implement the calculation of R_curve in meters (radius of curvature) #####
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Define fit polynomial in pixels
    left_fit_pix = np.polyfit(lefty, leftx, 2)
    right_fit_pix = np.polyfit(righty, rightx, 2)
    
    # Calculate the bottom pixel positions of left and right lanes
    left_pix = left_fit_pix[0]*y_eval**2 + left_fit_pix[1]*y_eval + left_fit_pix[2]
    right_pix = right_fit_pix[0]*y_eval**2 + right_fit_pix[1]*y_eval + right_fit_pix[2]
    
    # Calculate the lane center pixel position
    lane_center_pix = (right_pix + left_pix) / 2
    
    # Calculate the offset from center in meters
    center_offset = (binary_warped.shape[1]/2 - lane_center_pix) * xm_per_pix
    
    return left_curverad, right_curverad, center_offset


test_image8 = mpimg.imread('../test_images/test3.jpg')
undist8 = cal_undistort(test_image8)
threshold8 = threshold(undist8)
warped8 = perspective_transform(threshold8)
measure_curvature_real(warped8)


# ## Final Pipeline

# In[16]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(img):
    ### Pre-process input image
    # Undistorted image
    undist = cal_undistort(img)
    
    # Thresholded image
    thresholded = threshold(undist)
    
    # Birds-eye view image
    warped = perspective_transform(thresholded)
    
    out_img, left_fitx, right_fitx, ploty = fit_polynomial(warped)
    
    
    ### Calculate curve radius and center offset
    left_curverad, right_curverad, center_offset = measure_curvature_real(warped)
    left_curve = "%.2f m" % left_curverad
    right_curve = "%.2f m" % right_curverad
    center = "%.2f m" % center_offset
    
    ### Project lane on image
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))*255
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    newwarp, Minv = unwarp(color_warp)
    
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    #plt.imshow(result)
    
    # Put curve radii and center offset to output image
    cv2.putText(result, left_curve , (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), thickness=2)
    cv2.putText(result, right_curve , (950, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), thickness=2)
    cv2.putText(result, center, (550, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), thickness=2)
    
    '''
    ### Visualization
    output1 = cv2.resize(thresholded,(640, 360), interpolation = cv2.INTER_AREA)
    #output2 = cv2.resize(lanes,(640, 360), interpolation = cv2.INTER_AREA)
    # Create an array big enough to hold both images next to each other.
    vis = np.zeros((720, 1280+640, 3))
    # Copy both images into the composed image.
    vis[:720, :1280,:] = result
    vis[:360, 1280:1920,:] = output1
    #vis[360:720, 1280:1920,:] = output2
    '''
    
    return result

test_image = mpimg.imread('../test_images/test2.jpg')
output = process_image(test_image)
plt.imshow(output)


# ## Video testing

# In[18]:


project_output = 'project_video_output.mp4'

clip1 = VideoFileClip("../project_video.mp4")

project_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'project_clip.write_videofile(project_output, audio=False)')


# In[19]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(project_output))


# ## More challenging videos

# In[ ]:


challenge_output = 'challenge_video_output.mp4'

clip2 = VideoFileClip("../challenge_video.mp4")

challenge_clip = clip2.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')


# In[ ]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))


# In[ ]:


harder_challenge_output = 'harder_challenge_video_output.mp4'

clip3 = VideoFileClip("../harder_challenge_video.mp4")

harder_challenge_clip = clip3.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'harder_challenge_clip.write_videofile(harder_challenge_output, audio=False)')


# In[ ]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(harder_challenge_output))

