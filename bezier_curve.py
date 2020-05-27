### This is an altered version of the original code from user ImportanceOfBeingErnest's answer to a Stack Overflow question: https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib/50751932#50751932

import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
import skimage
from skimage import color # see the docs at scikit-image.org/
from skimage import measure
from PIL import Image  
import random as r
from random import random

OutputData = False
ShowFigures = True

bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)

## Rad ##
# The radius around the points at which the control points of the bezier curve sit. This number is relative to the distance between adjacent points and should hence be between 0 and 1. The larger the radius, the sharper the features of the curve.
## Edgy ##
# A parameter to determine the smoothness of the curve. If 0 the angle of the curve through each point will be the mean between the direction to adjacent points. The larger it gets, the more the angle will be determined only by one adjacent point. The curve hence gets "edgier".
## n or NumPoints ##
# The number of random points to use. Minimum 3. The more points used, the more feature rich the shapes can become; at the risk of creating overlaps or loops in the curve.
NumShapes = 1 # number of shapes you want to generate
NumPoints = 3 # number of points for the shape calculation
Rad = 0.5 # radius of the circle connecting two points (0 straight line) 
Edge = 0.01 # how pointy the radius of the circle connecting two points will be (0 smooth)

for numShapes in range(NumShapes):
	Rad = r.random() # float [0,1)
	Edge = r.uniform(0.0, 10.0) # float between [0,10)
	radStr = str(round(Rad,2)) # round up for file name
	edgeStr = str(round(Edge,2)) # round up for file name
	fileName = str(NumPoints)+'P_'+radStr+'R_'+edgeStr+'E_'+str(numShapes)

	## generate shape
	a = get_random_points(n=NumPoints, scale=1) 
	# a = np.array([[0,0], [0,1], [1,0], [1,1]])
	x,y, _ = get_bezier_curve(a,rad=Rad, edgy=Edge)

	## plot shape
	fig = plt.figure(0)
	ax = fig.subplots()
	ax.set_aspect("equal")
	plt.plot(x,y)
	ax.axis('off') # removes the axis to leave only the shape for contour
	if OutputData:
		fig.savefig('shapes/'+fileName+'.png')
	fig.canvas.draw()

	## convert shape to string, then to grayscale for contour
	data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
	data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	##image version of the original shape
	im = Image.frombytes("RGB", (fig.canvas.get_width_height()), data)
	if OutputData:
		im = im.save('images/'+fileName+'.png')
	# im.show()
	gray_image = color.rgb2gray(data)
	## save data string to text file
	if OutputData:
		imgSizeStr = '_Row'+str(gray_image.shape[0])+'_Col'+str(gray_image.shape[1])
		np.savetxt('grayDataPoints/'+fileName+imgSizeStr+'.txt', gray_image)
	smooth_contour = measure.find_contours(gray_image[::-1,:], 0.5)[0]

	## plot original version with axis
	fig1 = plt.figure(1)
	ax1 = fig1.subplots()
	ax1.set_aspect("equal")
	xPoints = a[:,0].tolist()
	yPoints = a[:,1].tolist()
	plt.plot(xPoints, yPoints, 'ro')
	plt.plot(x,y)
	if OutputData:
		fig1.savefig('points/'+fileName+'.png')
	fig1.canvas.draw()

	## plot contour version
	fig2 = plt.figure(2)
	ax2 = fig2.subplots()
	ax2.set_aspect("equal")
	plt.plot(smooth_contour[:, 1], smooth_contour[:, 0], linewidth=2, c='k')
	if OutputData:
		fig2.savefig('skimage_contour/'+fileName+'.png')
	fig2.canvas.draw()

	if ShowFigures:
		plt.show()

	## clear figures for next shape
	fig.clf()
	fig1.clf()
	fig2.clf()
