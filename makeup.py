from __future__ import division
import cv2
import numpy as np
from numpy.linalg import eig, inv
from scipy.interpolate import interp1d, splprep, splev
from scipy.interpolate import InterpolatedUnivariateSpline
from pylab import *
from skimage import color
from scipy import misc
import time


class makeup(object):
    """
    Class that handles application of color, and performs blending on image.

    Functions available for use:
        1. apply_lipstick: Applies lipstick on passed image of face.
        2. apply_liner: Applies black eyeliner on passed image of face.
        3. apply_blush: Applies blush on passed image of face.
        4. apply_eyebrow: Applies eyebrow on passed image of face.
        5. apply_foundation: Apply foundation passed image of face.
    """

    def __init__(self, img):
        """ Initiator method for class """
        self.red_l = 0
        self.green_l = 0
        self.blue_l = 0
        self.red_b = 0
        self.green_b = 0
        self.blue_b = 0
        self.image = img
        self.height, self.width = self.image.shape[:2]
        self.im_copy = self.image.copy()

        self.intensity = 0.8

        self.x = []
        self.y = []
        self.xleft=[]
        self.yleft=[]
        self.xright=[]
        self.yright=[]

    def __draw_curve(self, points, kind):
        """
        Draws a curve alone the given points by creating an interpolated path.
        """
        curvex = []
        curvey = []
        x_pts = list(np.asarray(points)[:,0])
        y_pts = list(np.asarray(points)[:,1])

        # Create the interpolated curve according to the x and y points.
        curve = interp1d(x_pts, y_pts, 'cubic')

        # Upper and lower curve are different in the order of points.
        if kind == 'upper':
            for i in np.arange(x_pts[0], x_pts[len(x_pts) - 1] + 1, 1):
                curvex.append(i)
                curvey.append(int(curve(i)))
        else:
            for i in np.arange(x_pts[len(x_pts) - 1] + 1, x_pts[0], 1):
                curvex.append(i)
                curvey.append(int(curve(i)))
        return curvex, curvey

    def __fill_lip_solid(self, outer, inner):
        """
        Fills solid colour inside two outlines.
        """
        outer_curve = zip(outer[0], outer[1])
        inner_curve = zip(inner[0], inner[1])
        points = []
        for point in outer_curve:
            points.append(np.array(point, dtype=np.int32))
        for point in inner_curve:
            points.append(np.array(point, dtype=np.int32))
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(self.image, [points], (self.red_l, self.green_l, self.blue_l))

        # Smoothen the color.
        img_base = np.zeros((self.height, self.width))
        cv2.fillPoly(img_base, [points], 1)
        img_mask = cv2.GaussianBlur(img_base, (51, 51), 0)
        img_blur_3d = np.ndarray([self.height, self.width, 3], dtype='float')
        img_blur_3d[:, :, 0] = img_mask
        img_blur_3d[:, :, 1] = img_mask
        img_blur_3d[:, :, 2] = img_mask
        self.im_copy = (img_blur_3d * self.image + (1 - img_blur_3d) * self.im_copy).astype('uint8')
        return

    def __draw_liner(self, eye, kind):
        """
        Draws eyeliner.
        """
        eye_x = []
        eye_y = []
        x_points = []
        y_points = []
        for point in eye:
            x_points.append(int(point.split()[0]))
            y_points.append(int(point.split()[1]))
        curve = interp1d(x_points, y_points, 'quadratic')
        for point in np.arange(x_points[0], x_points[len(x_points) - 1] + 1, 1):
            eye_x.append(point)
            eye_y.append(int(curve(point)))
        if kind == 'left':
            y_points[0] -= 1
            y_points[1] -= 1
            y_points[2] -= 1
            x_points[0] -= 5
            x_points[1] -= 1
            x_points[2] -= 1
            curve = interp1d(x_points, y_points, 'quadratic')
            count = 0
            for point in np.arange(x_points[len(x_points) - 1], x_points[0], -1):
                count += 1
                eye_x.append(point)
                if count < (len(x_points) / 2):
                    eye_y.append(int(curve(point)))
                elif count < (2 * len(x_points) / 3):
                    eye_y.append(int(curve(point)) - 1)
                elif count < (4 * len(x_points) / 5):
                    eye_y.append(int(curve(point)) - 2)
                else:
                    eye_y.append(int(curve(point)) - 3)
        elif kind == 'right':
            x_points[3] += 5
            x_points[2] += 1
            x_points[1] += 1
            y_points[3] -= 1
            y_points[2] -= 1
            y_points[1] -= 1
            curve = interp1d(x_points, y_points, 'quadratic')
            count = 0
            for point in np.arange(x_points[len(x_points) - 1], x_points[0], -1):
                count += 1
                eye_x.append(point)
                if count < (len(x_points) / 2):
                    eye_y.append(int(curve(point)))
                elif count < (2 * len(x_points) / 3):
                    eye_y.append(int(curve(point)) - 1)
                elif count < (4 * len(x_points) / 5):
                    eye_y.append(int(curve(point)) - 2)
                elif count:
                    eye_y.append(int(curve(point)) - 3)
        curve = zip(eye_x, eye_y)
        points = np.asarray(curve)
        cv2.fillPoly(self.im_copy, [points], 0)
        return

    def __get_points_lips(self, lips_points):
        """
        Get the points for the lips.
        """
        uol = []
        uil = []
        lol = []
        lil = []
        for i in range(0, 14, 2):
            uol.append([int(lips_points[i]), int(lips_points[i + 1])])
        for i in range(12, 24, 2):
            lol.append([int(lips_points[i]), int(lips_points[i + 1])])
        lol.append([int(lips_points[0]), int(lips_points[1])])
        for i in range(24, 34, 2):
            uil.append([int(lips_points[i]), int(lips_points[i + 1])])
        for i in range(32, 40, 2):
            lil.append([int(lips_points[i]), int(lips_points[i + 1])])
        lil.append([int(lips_points[24]), int(lips_points[25])])
        return uol, uil, lol, lil

    def __get_curves_lips(self, uol, uil, lol, lil):
        """
        Get the outlines of the lips.
        """
        uol_curve = self.__draw_curve(uol, 'upper')
        uil_curve = self.__draw_curve(uil, 'upper')
        lol_curve = self.__draw_curve(lol, 'lower')
        lil_curve = self.__draw_curve(lil, 'lower')
        return uol_curve, uil_curve, lol_curve, lil_curve

    def __fill_color(self, uol_c, uil_c, lol_c, lil_c):
        """
        Fill colour in lips.
        """
        self.__fill_lip_solid(uol_c, uil_c)
        self.__fill_lip_solid(lol_c, lil_c)
        return

    def __create_eye_liner(self, eyes_points):
        """
        Apply eyeliner.
        """
        left_eye = eyes_points[0].split('\n')
        right_eye = eyes_points[1].split('\n')
        right_eye = right_eye[0:4]
        self.__draw_liner(left_eye, 'left')
        self.__draw_liner(right_eye, 'right')
        return

    def __fitEllipse(self, x, y):
        """
        Given points of x and y, find out the most appropriate Ellipse
        """
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T, D)
        C = np.zeros([6, 6])
        C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
        E, V = eig(np.dot(inv(S), C))
        n = np.argmax(np.abs(E))
        a = V[:, n]
        return a

    def __ellipse_center(self, a):
        """
        Find out the center of ellipse.
        """
        b, c, d, f, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[0]
        num = b*b - a*c
        x0 = (c*d - b*f) / num
        y0 = (a*f - b*d) / num
        return np.array([x0, y0])

    def __ellipse_angle_of_rotation(self, a):
        """
        Find out how many angle should the ellipse rotate.
        """
        b, c, a = a[1]/2, a[2], a[0]
        return 0.5 * np.arctan(2*b / (a-c))

    def __ellipse_axis_length(self, a):
        """
        Find out the length of two axes.
        """
        b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        up = 2 * (a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
        down1 = (b*b - a*c) * ((c-a) * np.sqrt(1 + 4*b*b/((a-c)*(a-c))) - (c+a))
        down2 = (b*b - a*c) * ((a-c) * np.sqrt(1 + 4*b*b/((a-c)*(a-c))) - (c+a))
        res1 = np.sqrt(up / down1)
        res2 = np.sqrt(up / down2)
        return np.array([res1, res2])

    def __getEllipse(self, x, y):
        """
        Find out the most appropriate ellipse.
        """
        a = self.__fitEllipse(x, y)
        center = self.__ellipse_center(a)
        phi = self.__ellipse_angle_of_rotation(a)
        axes = self.__ellipse_axis_length(a)
        return (center[0], center[1]), (axes[0], axes[1]/1.3), phi

    def __univariate_plot(self, lx=[], ly=[]):
        """
        Interpolate with univariate spline.
        """
        unew = np.arange(lx[0], lx[-1]+1, 1)
        f2 = InterpolatedUnivariateSpline(lx, ly)
        return unew, f2(unew)

    def __inter_plot(self, lx=[], ly=[], k1='quadratic'):
        """
        Interpolate with interp1d.
        """
        unew = np.arange(lx[0], lx[-1]+1, 1)
        f2 = interp1d(lx, ly, kind=k1)
        return unew, f2(unew)

    def __getBoundaryPoints(self, x, y):
        """
        Given x and y, find out the boundary.
        """
        tck, u = splprep([x, y], s=0, per=1)
        unew = np.linspace(u.min(), u.max(), 10000)
        xnew, ynew = splev(unew, tck, der=0)
        tup = c_[xnew.astype(int), ynew.astype(int)].tolist()
        coord = list(set(tuple(map(tuple, tup))))
        coord = np.array([list(elem) for elem in coord])
        return coord[:, 0], coord[:, 1]

    def __getInteriorPoints(self, x, y):
        """
        Find out all points needed for interpolation.
        """
        intx = []
        inty = []

        def ext(a, b, i):
            a, b = int(a), int(b)
            intx.extend(np.arange(a, b, 1).tolist())
            temp = np.ones(b-a)*i
            inty.extend(temp.astype(int).tolist())
        x, y = np.array(x), np.array(y)
        xmin, xmax = amin(x), amax(x)
        xrang = np.arange(xmin, xmax+1, 1)
        for i in xrang:
            ylist = y[where(x == i)]
            ext(amin(ylist), amax(ylist), i)
        return intx, inty

    def __get_boundary_points(self, landmarks, flag):
        """
        Find out the boundary of blush.
        """
        if flag == 0:
            # Right Cheek
            r = (landmarks[15, 0] - landmarks[35, 0]) / 3.5
            center = (landmarks[15] + landmarks[35]) / 2.0
        elif flag == 1:
            # Left Cheek
            r = (landmarks[1, 0] - landmarks[31, 0]) / 3.5
            center = (landmarks[1] + landmarks[31]) / 2.0

        points_1 = [center[0] - r, center[1]]
        points_2 = [center[0], center[1] - r]
        points_3 = [center[0] + r, center[1]]
        points_4 = [center[0], center[1] + r]
        points_5 = points_1

        points = np.array([points_1, points_2, points_3, points_4, points_5])

        x, y = points[0:5, 0], points[0:5, 1]

        tck, u = splprep([x, y], s=0, per=1)
        unew = np.linspace(u.min(), u.max(), 1000)
        xnew, ynew = splev(unew, tck, der=0)
        tup = c_[xnew.astype(int), ynew.astype(int)].tolist()
        coord = list(set(tuple(map(tuple, tup))))
        coord = np.array([list(elem) for elem in coord])
        return np.array(coord[:, 0], dtype=np.int32), np.array(coord[:, 1], dtype=np.int32)

    def __blush(self, x_right, y_right, x_left, y_left):

        intensity = 0.3
        # Create blush shape
        mask = np.zeros((self.height, self.width))
        cv2.fillConvexPoly(mask, np.array(c_[x_right, y_right], dtype='int32'), 1)
        cv2.fillConvexPoly(mask, np.array(c_[x_left, y_left], dtype='int32'), 1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0) * intensity

        # Add blush color to image
        # val = color.rgb2lab((self.im_copy / 255.))
        val = cv2.cvtColor(self.im_copy, cv2.COLOR_RGB2LAB).astype(float)
        val[:, :, 0] = val[:, :, 0] / 255. * 100.
        val[:, :, 1] = val[:, :, 1] - 128.
        val[:, :, 2] = val[:, :, 2] - 128.
        LAB = color.rgb2lab(np.array((self.red_b / 255., self.green_b / 255., self.blue_b / 255.)).reshape(1, 1, 3)).reshape(3,)

        mean_val = np.mean(np.mean(val, axis=0), axis = 0)
        mask = np.array([mask,mask,mask])
        mask = np.transpose(mask, (1,2,0))
        lab = np.multiply((LAB - mean_val), mask)

        val[:, :, 0] = np.clip(val[:, :, 0] + lab[:,:,0], 0, 100)
        val[:, :, 1] = np.clip(val[:, :, 1] + lab[:,:,1], -127, 128)
        val[:, :, 2] = np.clip(val[:, :, 2] + lab[:,:,2], -127, 128)

        self.im_copy = (color.lab2rgb(val) * 255).astype(np.uint8)
        # val[:, :, 0] = (np.clip(val[:, :, 0] + lab[:,:,0], 0, 100) / 100 * 255).astype(np.uint8)
        # val[:, :, 1] = (np.clip(val[:, :, 1] + lab[:,:,1], -127, 128) + 127).astype(np.uint8)
        # val[:, :, 2] = (np.clip(val[:, :, 2] + lab[:,:,2], -127, 128) + 127).astype(np.uint8)

        # self.im_copy = cv2.cvtColor(val, cv2.COLOR_LAB2RGB)

    def __get_lips(self, landmarks, flag=None):
        """
        Find out the landmarks corresponding to lips.
        """
        if landmarks is None:
            return None
        lips = ""
        for point in landmarks[48:]:
            lips += str(point).replace('[', '').replace(']', '') + '\n'
        return lips

    def __get_upper_eyelids(self, landmarks, flag=None):
        """
        Find out landmarks corresponding to upper eyes.
        """
        if landmarks is None:
            return None
        liner = ""
        for point in landmarks[36:40]:
            liner += str(point).replace('[', '').replace(']', '') + '\n'
        liner += '\n'
        for point in landmarks[42:46]:
            liner += str(point).replace('[', '').replace(']', '') + '\n'
        return liner

    def apply_lipstick(self, landmarks, rlips, glips, blips):
        """
        Applies lipstick on an input image.
        ___________________________________
        Inputs:
            1. Landmarks of the face.
            2. Colors of lipstick in the order of r, g, b.
        Output:
            1. The face applied with lipstick.

        """

        self.red_l = rlips
        self.green_l = glips
        self.blue_l = blips
        lips = self.__get_lips(landmarks)
        lips = list([point.split() for point in lips.split('\n')])
        lips_points = [item for sublist in lips for item in sublist]
        uol, uil, lol, lil = self.__get_points_lips(lips_points)
        uol_c, uil_c, lol_c, lil_c = self.__get_curves_lips(uol, uil, lol, lil)
        self.__fill_color(uol_c, uil_c, lol_c, lil_c)
        return self.im_copy

    def apply_liner(self, landmarks):
        """
        Applies black liner on the input image.
        ___________________________________
        Input:
            1. Landmarks of the face.
        Output:
            1. The face applied with eyeliner.
        """
        liner = self.__get_upper_eyelids(landmarks)
        eyes_points = liner.split('\n\n')
        self.__create_eye_liner(eyes_points)
        return self.im_copy

    def apply_foundation(self, landmarks):
        """
        Applies foundation on the input image.
        ___________________________________
        Input:
            1. Landmarks of the face.
        Output:
            1. The face applied with foundation.
        """

        # R, G, B = (200., 121., 46.)
        R, G, B = (234., 135., 103.)
        inten = 0.6

        # Get points of face.
        fileface = landmarks[0:17]
        pointsface = np.floor(fileface)
        point_face_x = np.array((pointsface[:][:, 0]))
        point_face_y = np.array(pointsface[:][:, 1])

        # Get points of lips.
        file = landmarks[48:68]
        points = np.floor(file)
        point_out_x = np.array((points[:int(len(points)/2)][:, 0]))
        point_out_y = np.array(points[:int(len(points)/2)][:, 1])

        # Get points of eyes.
        fileeye = landmarks[36:48]
        pointseye = np.floor(fileeye)
        eye_point_down_x = np.array(pointseye[:4][:, 0])
        eye_point_down_y = np.array(pointseye[:4][:, 1])
        eye_point_up_x = (pointseye[3:6][:, 0]).tolist()
        eye_point_up_y = (pointseye[3:6][:, 1]).tolist()
        eye_point_up_x.append(pointseye[0, 0])
        eye_point_up_y.append(pointseye[0, 1])
        eye_point_up_x = np.array(eye_point_up_x)
        eye_point_up_y = np.array(eye_point_up_y)
        eye_point_down_x_right = np.array(pointseye[6:10][:, 0])
        eye_point_down_y_right = np.array(pointseye[6:10][:, 1])
        eye_point_up_x_right = (pointseye[9:12][:, 0]).tolist()
        eye_point_up_y_right = (pointseye[9:12][:, 1]).tolist()
        eye_point_up_x_right.append(pointseye[6, 0])
        eye_point_up_y_right.append(pointseye[6, 1])
        eye_point_up_x_right = np.array(eye_point_up_x_right)
        eye_point_up_y_right = np.array(eye_point_up_y_right)

        x_face = []
        y_face = []
        x_aux = []
        y_aux = []

        # Get lower face from landmarks.
        lower_face = self.__univariate_plot(point_face_x[:], point_face_y[:])
        x_face.extend(lower_face[0][::-1])
        y_face.extend(lower_face[1][::-1])

        # Get upper face from approximation.
        (centerx, centery), (axesx, axesy), angel = self.__getEllipse(point_face_x, point_face_y)
        centerpt = (int(centerx), int(centery))
        axeslen = (int(axesx), int(axesy*1.2))
        ellippoints = np.floor(cv2.ellipse2Poly(centerpt, axeslen, int(angel), 180, 360, 1))
        ellipx = ellippoints[:, 0].tolist()
        ellipy = ellippoints[:, 1].tolist()
        x_face.extend(ellipx)
        y_face.extend(ellipy)
        x_face.append(x_face[0])
        y_face.append(y_face[0])
        x_face, y_face = self.__getBoundaryPoints(x_face, y_face)
        x, y = self.__getInteriorPoints(x_face, y_face)

        # Remove lips from face mask.
        l_u_l = self.__inter_plot(point_out_x[:4], point_out_y[:4])
        l_u_r = self.__inter_plot(point_out_x[3:7], point_out_y[3:7])
        l_l = self.__inter_plot([point_out_x[0]]+point_out_x[6:][::-1].tolist(), [point_out_y[0]]+point_out_y[6:][::-1].tolist(), 'cubic')
        lipinteriorx, lipinteriory = self.__getInteriorPoints(l_u_l[0].tolist() + l_u_r[0].tolist() + l_l[0].tolist(), l_u_l[1].tolist() + l_u_r[1].tolist() + l_l[1].tolist())
        x_aux.extend(lipinteriorx)
        y_aux.extend(lipinteriory)

        # Remove eyes from face mask.
        e_l_l = self.__inter_plot(eye_point_down_x[:], eye_point_down_y[:], 'cubic')
        e_u_l = self.__inter_plot(eye_point_up_x[:], eye_point_up_y[:], 'cubic')
        lefteyex, lefteyey = self.__getInteriorPoints(e_l_l[0].tolist() + e_u_l[0].tolist(), e_l_l[1].tolist() + e_u_l[1].tolist())
        x_aux.extend(lefteyex)
        y_aux.extend(lefteyey)
        e_l_r = self.__inter_plot(eye_point_down_x_right[:], eye_point_down_y_right[:], 'cubic')
        e_u_r = self.__inter_plot(eye_point_up_x_right[:], eye_point_up_y_right[:], 'cubic')
        righteyex, righteyey = self.__getInteriorPoints(e_l_r[0].tolist() + e_u_r[0].tolist(), e_l_r[1].tolist() + e_u_r[1].tolist())
        x_aux.extend(righteyex)
        y_aux.extend(righteyey)

        val = (color.rgb2lab((self.im_copy[x, y]/255.).reshape(len(x), 1, 3))
                    .reshape(len(x), 3))
        vallips = (color.rgb2lab((self.im_copy[x_aux, y_aux]/255.).reshape(len(x_aux), 1, 3))
                        .reshape(len(x_aux), 3))
        L = (sum(val[:, 0])-sum(vallips[:, 0]))/(len(val[:, 0])-len(vallips[:, 0]))
        A = (sum(val[:, 1])-sum(vallips[:, 1]))/(len(val[:, 1])-len(vallips[:, 1]))
        bB = (sum(val[:, 2])-sum(vallips[:, 2]))/(len(val[:, 2])-len(vallips[:, 2]))

        L1, A1, B1 = color.rgb2lab(np.array((R/255., G/255., B/255.)).reshape(1, 1, 3)).reshape(3,)
        val[:, 0] += (L1-L) * inten
        val[:, 1] += (A1-A) * inten
        val[:, 2] += (B1-bB) * inten

        self.im_copy[x, y] = color.lab2rgb(val.reshape(len(x), 1, 3)).reshape(len(x), 3) * 255
        return self.im_copy

    def apply_blush(self, landmarks, R, G, B):
        """
        Applies blush on the input image.
        ___________________________________
        Input:
            1. Landmarks of the face.
            2. Color of blush in the order of r, g, b.
        Output:
            1. The face applied with blush.
        """

        # Find Blush Loacations
        x_right, y_right = self.__get_boundary_points(landmarks, 0)
        x_left, y_left = self.__get_boundary_points(landmarks, 1)

        # Apply Blush
        self.red_b = R
        self.green_b = G
        self.blue_b = B
        self.__blush(x_right, y_right, x_left, y_left)

        return self.im_copy

    def apply_eyebrow(self, landmark):
        """
        Applies eyebrow on the input image.
        ___________________________________
        Input:
            1. Landmarks of the face.
        Output:
            1. The face applied with eyebrow.
        """
        # right eyebrow
        pts1 = np.array(landmark[17:22], np.int32)

        # rescale
        right_eye = misc.imread('./data/right_eyebrow.png')
        scale = float(pts1[4][0] - pts1[0][0]) / right_eye.shape[1]
        right_eye = cv2.resize(right_eye, (0, 0), fx=scale, fy=scale)

        # find location
        x_offset = pts1[0][0]
        y_offset = pts1[0][1]
        y1, y2 = y_offset - right_eye.shape[0] + 5, y_offset + 5
        x1, x2 = x_offset, x_offset + right_eye.shape[1]

        alpha_s = right_eye[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        # apply eyebrow
        for c in range(0, 3):
            self.im_copy[y1:y2, x1:x2, c] = (alpha_s * right_eye[:, :, c] +
                                      alpha_l * self.im_copy[y1:y2, x1:x2, c])

        # left eyebrow
        pts2 = np.array(landmark[22:27], np.int32)

        # rescale
        left_eye = misc.imread('./data/left_eyebrow.png')
        scale = float(pts2[4][0] - pts2[0][0]) / left_eye.shape[1]
        left_eye = cv2.resize(left_eye, (0, 0), fx=scale, fy=scale)

        # find location
        x_offset = pts2[0][0]
        y_offset = pts2[0][1]
        y1, y2 = y_offset - left_eye.shape[0] + 5, y_offset + 5
        x1, x2 = x_offset, x_offset + left_eye.shape[1]

        alpha_s = left_eye[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        # apply eyebrow
        for c in range(0, 3):
            self.im_copy[y1:y2, x1:x2, c] = (alpha_s * left_eye[:, :, c] +
                                      alpha_l * self.im_copy[y1:y2, x1:x2, c])
        return self.im_copy

    def __inter(self, lx=[], ly=[], k1='quadratic'):
        unew = np.arange(lx[0], lx[-1]+1, 1)
        f2 = interp1d(lx, ly, kind=k1)
        return (f2,unew)

    def __ext(self, a, b, i):
        self.x.extend(arange(a,b,1).tolist())
        if(b-a==1):
            self.y.extend((ones(b-a)*i).tolist())
        else:
            self.y.extend((ones(b-a+1)*i).tolist())

    def __extleft(self, a, b, i):
    	self.xleft.extend(arange(a,b,1).tolist())
    	if(b-a==1):
    		self.yleft.extend((ones(b-a)*i).tolist())
    	else:
    		self.yleft.extend((ones(b-a+1)*i).tolist())

    def __extright(self, a, b, i):
    	self.xright.extend(arange(a,b,1).tolist())
    	if(b-a==1):
    		self.yright.extend((ones(b-a)*i).tolist())
    	else:
    		self.yright.extend((ones(b-a+1)*i).tolist())

    def apply_eyeshadow(self, landmarks_array, R, G, B):

        lower_left_end = 5
        upper_left_end = 11
        lower_right_end = 16
        upper_right_end = 22

        eye_left = landmarks_array[36: 40, :]
        eye_left_end = np.array([landmarks_array[39, :]])
        eyebrow_left = landmarks_array[17 : 22, :]
        eyebrow_left_start = np.array([landmarks_array[17, :]])

        left = np.concatenate((eyebrow_left_start, eye_left, eyebrow_left, eye_left_end), axis=0)
        if(left[9, 0] > left[10, 0]):
            left[9, 0] = left[10, 0] - 5

        eye_right = landmarks_array[42: 46, :]
        eye_right_start = np.array([landmarks_array[42, :]])
        eyebrow_right = landmarks_array[22 : 27, :]
        eyebrow_right_end = np.array([landmarks_array[26 , :]])

        right = np.concatenate((eye_right, eyebrow_right_end, eye_right_start, eyebrow_right), axis = 0)
        if(right[6, 0] < right[5, 0]):
            right[6, 0] = right[5, 0] + 5

        points = np.concatenate((left, right), axis = 0)

        point_down_x = np.array((points[:lower_left_end][:,0]))
        point_down_y = np.array(points[:lower_left_end][:,1])
        point_up_x = np.array(points[lower_left_end:upper_left_end][:,0])
        point_up_y = np.array(points[lower_left_end:upper_left_end][:,1])
        point_down_x_right = np.array((points[upper_left_end:lower_right_end][:,0]))
        point_down_y_right = np.array(points[upper_left_end:lower_right_end][:,1])
        point_up_x_right = np.array((points[lower_right_end:upper_right_end][:,0]))
        point_up_y_right = np.array(points[lower_right_end:upper_right_end][:,1])

        point_down_y_max = max(point_down_y)
        point_up_y_min = min(point_up_y)
        offset_left = point_down_y_max - point_up_y_min

        point_up_y[0] += offset_left * 0.3
        point_up_y[1] += offset_left * 0.3
        point_up_y[2] += offset_left * 0.15
        point_up_y[3] += offset_left * 0.1
        point_up_y[4] += offset_left * 0.3
        point_down_y[0] += offset_left * 0.3

        point_down_y_right_max = max(point_down_y_right)
        point_up_y_right_min = min(point_up_y_right)
        offset_right = point_down_y_right_max - point_up_y_right_min

        point_up_y_right[-1] += offset_right * 0.3
        point_up_y_right[1] += offset_right * 0.3
        point_up_y_right[2] += offset_right * 0.1
        point_up_y_right[3] += offset_right * 0.15
        point_up_y_right[4] += offset_right * 0.3
        point_down_y_right[-1] += offset_right * 0.3

        im = self.im_copy
        im2 = im.copy()
        height,width = im.shape[:2]

        # bound the convex poly
        l_l = self.__inter(point_down_x[:],point_down_y[:],'cubic')
        u_l = self.__inter(point_up_x[:],point_up_y[:],'cubic')
        l_r = self.__inter(point_down_x_right[:],point_down_y_right[:],'cubic')
        u_r = self.__inter(point_up_x_right[:],point_up_y_right[:],'cubic')

        for i in range(int(l_l[1][0]),int(l_l[1][-1]+1)):
            self.__ext(u_l[0](i),l_l[0](i)+1,i)
            self.__extleft(u_l[0](i),l_l[0](i)+1,i)

        for i in range(int(l_r[1][0]),int(l_r[1][-1]+1)):
            self.__ext(u_r[0](i),l_r[0](i)+1,i)
            self.__extright(u_r[0](i),l_r[0](i)+1,i)

        # add color to eyeshadow area
    	val = color.rgb2lab((im[self.x,self.y]/255.).reshape(len(self.x),1,3)).reshape(len(self.x),3)
        L, A, bB = mean(val[:,0]), mean(val[:,1]), mean(val[:,2])

        rgbmean = (im[self.x,self.y])
        rmean, gmean, bmean = mean(rgbmean[:,0]), mean(rgbmean[:,1]), mean(rgbmean[:,2])

        L,A,bB = color.rgb2lab(np.array((rmean/255.,gmean/255.,bmean/255.)).reshape(1,1,3)).reshape(3,)
        L1,A1,B1 = color.rgb2lab(np.array((R/255.,G/255.,B/255.)).reshape(1,1,3)).reshape(3,)

        #compare the difference between the original and goal color
        val[:,0] += (L1-L) * self.intensity
        val[:,1] += (A1-A) * self.intensity
        val[:,2] += (B1-bB) * self.intensity

        image_blank = np.zeros([height, width, 3])
        image_blank[self.x,self.y] = color.lab2rgb(val.reshape(len(self.x),1,3)).reshape(len(self.x),3)*255

        original = color.rgb2lab((im[self.x,self.y]*0/255.).reshape(len(self.x),1,3)).reshape(len(self.x),3)
        tobeadded = color.rgb2lab((image_blank[self.x,self.y]/255.).reshape(len(self.x),1,3)).reshape(len(self.x),3)
        original += tobeadded
        im[self.x,self.y] = color.lab2rgb(original.reshape(len(self.x),1,3)).reshape(len(self.x),3)*255

        # Blur Filter
        filter = np.zeros((height, width))
        cv2.fillConvexPoly(filter,np.array(c_[self.yleft, self.xleft], dtype = 'int32'),1)
        cv2.fillConvexPoly(filter,np.array(c_[self.yright, self.xright], dtype = 'int32'),1)
        filter = cv2.GaussianBlur(filter,(31,31),0)

        # Erosion to reduce blur size
        kernel = np.ones((12, 12), np.uint8)
        filter = cv2.erode(filter, kernel, iterations = 1)
        alpha=np.zeros([height,width,3],dtype='float64')
        alpha[:,:,0], alpha[:,:,1], alpha[:,:,2] = filter, filter, filter

        return (alpha * im + (1 - alpha) * im2).astype('uint8')
        # return (alpha * im + (1 - alpha) * im2)
    	# imshow((alpha * im + (1 - alpha) * im2).astype('uint8'))

    def apply_makeup(self, landmarks):
        self.im_copy = self.apply_lipstick(landmarks, 170, 10, 30)
        # self.im_copy = self.apply_foundation(landmarks)
        # self.im_copy = self.apply_blush(landmarks, 223., 91., 111.)
        self.im_copy = self.apply_eyebrow(landmarks)
        self.im_copy = self.apply_liner(landmarks)
        # self.im_copy = self.apply_eyeshadow(landmarks, 102, 0, 51)
        return self.im_copy

    def apply_makeup_all(self, landmarks):
        self.im_copy = self.apply_lipstick(landmarks, 170, 10, 30)
        self.im_copy = self.apply_foundation(landmarks)
        self.im_copy = self.apply_blush(landmarks, 223., 91., 111.)
        self.im_copy = self.apply_eyebrow(landmarks)
        self.im_copy = self.apply_liner(landmarks)
        self.im_copy = self.apply_eyeshadow(landmarks, 102, 0, 51)
        return self.im_copy
