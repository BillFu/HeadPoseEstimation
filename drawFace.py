import cv2
import numpy as np


def drawPolyline(img, shapes, start, end, isClosed=False):
    points = []
    for i in range(start, end + 1):
        point = [shapes.part(i).x, shapes.part(i).y]
        points.append(point)
    points = np.array(points, dtype=np.int32)
    cv2.polylines(img, [points], isClosed, (255, 80, 0),
                  thickness=1, lineType=cv2.LINE_8)

def draw(img, shapes):
    drawPolyline(img, shapes, 0, 16)
    drawPolyline(img, shapes, 17, 21)
    drawPolyline(img, shapes, 22, 26)
    drawPolyline(img, shapes, 27, 30)
    drawPolyline(img, shapes, 30, 35, True)
    drawPolyline(img, shapes, 36, 41, True)
    drawPolyline(img, shapes, 42, 47, True)
    drawPolyline(img, shapes, 48, 59, True)
    drawPolyline(img, shapes, 60, 67, True)


def drawPnP2DRefPts(canvas_img, pointsArray, point_size, point_color, thickness):
    num_point = pointsArray.shape[0]
    for i in range(num_point):
        point_i = pointsArray[i, :]
        point = (int(point_i[0]), int(point_i[1]))
        cv2.circle(canvas_img, point, point_size, point_color, thickness)


def drawPnP2DRefOnePt(canvas_img, point_i, point_size, point_color, thickness):
    point = (int(point_i[0]), int(point_i[1]))
    cv2.circle(canvas_img, point, point_size, point_color, thickness)
