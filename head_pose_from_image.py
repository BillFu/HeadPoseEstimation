#!/usr/bin/env python3

import os
import cv2
import sys
import dlib
import numpy as np

from drawFace import drawPnP2DRefPts, drawPnP2DRefOnePt
import reference_world as world

PREDICTOR_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")

if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] USE models/downloader.sh to download the predictor")
    sys.exit()


def main(inImgFile, annoFile):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    image_bgr = cv2.imread(inImgFile)

    faces = detector(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), 0)
    face3Dmodel = world.ref3DModel()

    face = faces[0]
    shape = predictor(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), face)

    # draw(image_bgr, shape)

    refImgPts = world.ref2dImagePoints(shape)

    point_size = 4
    blue_color = (255, 0, 0)  # BGR
    thickness = 4
    drawPnP2DRefPts(image_bgr, refImgPts, point_size, blue_color, thickness)

    red_color = (0, 0, 255)  # BGR
    point_i = refImgPts[0, :]
    drawPnP2DRefOnePt(image_bgr, point_i, 20, red_color, thickness)

    height, width, channel = image_bgr.shape
    # focalLength = args.focal * width
    focalLength = width
    cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))

    mdists = np.zeros((4, 1), dtype=np.float64)

    # calculate rotation and translation vector using solvePnP
    success, rotationVector, translationVector = cv2.solvePnP(
        face3Dmodel, refImgPts, cameraMatrix, mdists)

    noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
    noseEndPoint2D, jacobian = cv2.projectPoints(
        noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

    # draw nose line
    p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
    p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
    cv2.line(image_bgr, p1, p2, (110, 220, 0),
             thickness=2, lineType=cv2.LINE_AA)

    # calculating angle
    rmat, jac = cv2.Rodrigues(rotationVector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles[0], angles[1], angles[2]

    gaze = "Looking: "
    if yaw < -10:
        gaze += "Left"
    elif yaw > 10:
        gaze += "Right"
    else:
        gaze += "Forward"

    cv2.putText(image_bgr, gaze, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 80), 2)
    cv2.putText(image_bgr, "pitch: " + str(np.round(pitch, 2)), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.putText(image_bgr, "yaw: " + str(np.round(yaw, 2)), (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.putText(image_bgr, "roll: " + str(np.round(roll, 2)), (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    cv2.imwrite(annoFile, image_bgr)


if __name__ == "__main__":
    """
    inImgFile = "./image/front/cross_1.jpg"
    annoFile = "./result/front/cross_1.png"
    main(inImgFile, annoFile)
    """

    inImgFile = "./image/profile/cross_4.jpg"
    annoFile = "./result/profile/cross_4.png"
    main(inImgFile, annoFile)
