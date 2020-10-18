import cv2
import time
import numpy as np
import math
from math import *
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy import interpolate


plt.ion()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.axes.set_xlim3d(-50,50)
ax.axes.set_ylim3d(-50,50)
ax.axes.set_zlim3d(-50,50)

FocalLength = (247 * 24)/11
FOV = 60
FOVH = 45
cap = cv2.VideoCapture(2)
low_yellow = (50/400*180, 40/100*255, 100)
high_yellow = (70/400*180, 100/100*255, 255)

locationmap = np.zeros((200,200,3), np.uint8)

out = cv2.VideoWriter('output4.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,960))

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)


def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

def contour_segmentation(im):
    alpha = 1.5 # Contrast control (1.0-3.0)
    beta = 10 # Brightness control (0-100)

    im = cv2.convertScaleAbs(im, alpha=alpha, beta=beta)

    imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # b&w

    imhsv = cv2.GaussianBlur(imhsv,(5,5),0)
    imhsv = cv2.blur(imhsv,(10,10))

    mask = cv2.inRange(imhsv, low_yellow, high_yellow)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    result = cv2.bitwise_and(imhsv, imhsv, mask=mask)
    
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    cv2.imshow("oh",result)

    ret,thresh1 = cv2.threshold(result,0,255,0)
    cnts, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    return cnts
    #contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #make contours around them
    #return contours

x1=list(range(5))
y1=list(range(5))
z1=list(range(5))
linecolor=[1,1,1,1,1]
times=[time.time()-1]
magnitudes=[0]
velocities=[0,0]
accelerations=[0,0,0]
img=0
while(cap.isOpened()):
    times.append(time.time())
    ret, frame = cap.read()
    ax.cla()
    
    ax.scatter3D([-50,50], [-50,50], [-50,50]);
    #cv2.imshow('original',frame)
    contours = contour_segmentation(frame.copy())
    # cv2.drawContours(frame, contours, -1, (0,255,0), 5) 
    locationmap = np.zeros((200,200,3), np.uint8)
    cv2.circle(locationmap, (100,200),10,(255,0,0),4)
    if contours:
        for contour in [max(contours,key=cv2.contourArea)]:
            area = cv2.contourArea(contour)
            if area<100 or (cv2.contourArea(cv2.convexHull(contour)) - area)/area > .5:
                continue
            (x,y),radius = cv2.minEnclosingCircle(contour)
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(frame,center,radius,(0,255,0),3)
            inches = distance_to_camera(2.7, FocalLength, radius*2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"{inches}in", (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
            angle = FOV * (x-320)/320
            rad = angle/360 * 2*math.pi
            angleh = FOVH * (y-240)/240
            radh = angleh/360 * 2*math.pi

            x1.append(inches * sin (rad) * cos(radh))
            z1.append(-inches * sin (radh))
            y1.append(inches * cos (rad) * cos(radh))
            

            if len(x1)>6:
                x1=x1[1:7]
                y1=y1[1:7]
                z1=z1[1:7]
                linecolor=linecolor[1:9]
            if len(times)>5:
                times=times[1:5]

            magnitude = math.sqrt((x1[-1]-x1[-2])**2+(y1[-1]-y1[-2])**2+(z1[-1]-z1[-2])**2)
            magnitudes.append(magnitude)
            velocity = magnitude/(times[-1]-times[-2])
            velocities.append(velocity)
            acceleration=(velocities[-1]-velocities[-2])/(times[-1]-times[-2])
            accelerations.append(acceleration)
            jerk = (accelerations[-1]-accelerations[-2])/(times[-1]-times[-2])
            linecolor.append(velocity/100)

            print(f"vel {velocity}in/s")
            if abs(acceleration)>500:
                del x1[-1]
                del y1[-1]
                del z1[-1]
                del times[-1]
                del velocities[-1]
                del magnitudes[-1]
            cv2.circle(locationmap, (int(math.tan(rad)*inches)+100,200-int(inches)),5,(0,255,0),3)

            #ax.plot3D(x1,y1,z1, c="red");
            tck, u = interpolate.splprep([x1,y1,z1], s=len(x1)-4)
            x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
            u_fine = np.linspace(0,1,100)
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            #ax.plot3D(x_fine,y_fine,z_fine, c="green");
            
            points = np.array([x_fine, y_fine, z_fine]).transpose().reshape(-1,1,3)
            # set up a list of segments
            segs = np.concatenate([points[:-1],points[1:]],axis=1)

            # make the collection of segments
            lc = Line3DCollection(segs, cmap=plt.get_cmap('cool'), norm=plt.Normalize(0, 1))
            colors = [[color]*math.floor(100/len(linecolor)) for color in linecolor]
            colors = [j for i in colors for j in i]
            colors = colors + (100-len(colors))*[colors[-1]]
            # colors = linecolor
            colors = [math.sqrt(e)/2 for e in colors]
            print(np.mean(colors))

            tck = interpolate.splrep(list(range(len(colors))), colors)
            colors=[interpolate.splev(e, tck) for e in range(len(colors))]
            lc.set_array(np.array(colors)) # color the segments by our parameter
            ax.add_collection3d(lc)

            #plt.draw() 
            plt.pause(0.02)
            #graph_image = cv2.cvtColor(np.array(fig.canvas.get_renderer()._renderer),cv2.COLOR_RGB2BGR) 
            #fig.canvas.draw()
            plt.show()
            # convert canvas to image
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                    sep='')
            img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # img is rgb, convert to opencv's default bgr
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            #print(f"{inches}in, {angle} degrees")
    #cv2.imshow('thing',img)
    #cv2.waitKey(0)
    if not isinstance(img,int):
        frame = vconcat_resize_min([img,frame])
        #print(frame.shape)
        #cv2.imshow('location',locationmap)
        cv2.imshow('frame',frame)
        out.write(frame)
    #time.sleep(.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()