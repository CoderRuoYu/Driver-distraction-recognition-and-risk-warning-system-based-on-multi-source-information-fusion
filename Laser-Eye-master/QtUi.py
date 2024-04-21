from QtTest import Ui_CamShow
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import pymysql
import sys
sys.path.append("..")
import pymysql
import traceback
from service.head_pose import HeadPoseEstimator
from service.face_alignment import CoordinateAlignmentModel
from service.face_detector import MxnetDetectionModel
from service.iris_localization import IrisLocalizationModel
import cv2
import numpy as np
from numpy import sin, cos, pi, arctan
from numpy.linalg import norm
from threading import Thread

SIN_LEFT_THETA = 2 * sin(pi / 4)
SIN_UP_THETA = sin(pi / 6)
from pynput import keyboard

class CamShow(QMainWindow, Ui_CamShow):
    def on_press(key):
        try:
            print('alphanumeric key {0} pressed'.format(key.char))
        except AttributeError:
            print('special key {0} pressed'.format(key))

    def on_release(key):
        print('{0} released'.format(key))
        # 将这里的逻辑替换成你要处理的逻辑即可，比如Key.space之类
        if key == keyboard.Key.esc:
            return False

    def calculate_3d_gaze(frame, poi, scale=256):
        starts, ends, pupils, centers = poi

        eye_length = norm(starts - ends, axis=1)
        ic_distance = norm(pupils - centers, axis=1)
        zc_distance = norm(pupils - starts, axis=1)

        s0 = (starts[:, 1] - ends[:, 1]) * pupils[:, 0]
        s1 = (starts[:, 0] - ends[:, 0]) * pupils[:, 1]
        s2 = starts[:, 0] * ends[:, 1]
        s3 = starts[:, 1] * ends[:, 0]

        delta_y = (s0 - s1 + s2 - s3) / eye_length / 2
        delta_x = np.sqrt(abs(ic_distance ** 2 - delta_y ** 2))

        delta = np.array((delta_x * SIN_LEFT_THETA,
                          delta_y * SIN_UP_THETA))
        delta /= eye_length
        theta, pha = np.arcsin(delta)

        # print(f"THETA:{180 * theta / pi}, PHA:{180 * pha / pi}")
        # delta[0, abs(theta) < 0.1] = 0
        # delta[1, abs(pha) < 0.03] = 0

        inv_judge = zc_distance ** 2 - delta_y ** 2 < eye_length ** 2 / 4

        delta[0, inv_judge] *= -1
        theta[inv_judge] *= -1
        delta *= scale

        # cv2.circle(frame, tuple(pupil.astype(int)), 2, (0, 255, 255), -1)
        # cv2.circle(frame, tuple(center.astype(int)), 1, (0, 0, 255), -1)

        return theta, pha, delta.T

    def draw_sticker(src, offset, pupils, landmarks,
                     blink_thd=0.22,
                     arrow_color=(0, 125, 255), copy=False):
        if copy:
            src = src.copy()

        left_eye_hight = landmarks[33, 1] - landmarks[40, 1]
        left_eye_width = landmarks[39, 0] - landmarks[35, 0]

        right_eye_hight = landmarks[87, 1] - landmarks[94, 1]
        right_eye_width = landmarks[93, 0] - landmarks[89, 0]

        for mark in landmarks.reshape(-1, 2).astype(int):
            cv2.circle(src, tuple(mark), radius=1,
                       color=(0, 0, 255), thickness=-1)

        if left_eye_hight / left_eye_width > blink_thd:
            cv2.arrowedLine(src, tuple(pupils[0].astype(int)),
                            tuple((offset + pupils[0]).astype(int)), arrow_color, 2)

        if right_eye_hight / right_eye_width > blink_thd:
            cv2.arrowedLine(src, tuple(pupils[1].astype(int)),
                            tuple((offset + pupils[1]).astype(int)), arrow_color, 2)

        return src



    def __init__(self, parent=None):
        super(CamShow, self).__init__(parent)
        self.setupUi(self)
        self.timer_camera = QTimer()  # 初始化定时器
        self.cap = cv2.VideoCapture()  # 初始化摄像头
        self.timer_camera.timeout.connect(self.show_camera)  # 当计时器到了一定的时间调用show_camera函数
        self.timer_camera.timeout.connect(self.load)
        self.open()

    def open(self):
        self.cap.open(0)
        self.timer_camera.start(30)

    def show_camera(self):
        gpu_ctx = -1
        n = 1
        fd = MxnetDetectionModel("weights/16and32", 0, .6, gpu=gpu_ctx)
        fa = CoordinateAlignmentModel('weights/2d106det', 0, gpu=gpu_ctx)
        gs = IrisLocalizationModel("weights/iris_landmark.tflite")
        hp = HeadPoseEstimator("weights/object_points.npy", self.cap.get(3), self.cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (960, 540))
        ret, frame = self.cap.read()
        if not ret:
            return

        try:
            bboxes = fd.detect(frame)
        except:
            traceback.print_exc()
            pass
        # poster = Thread(target=fd.workflow_postprocess)
        # poster.start()

        # infer = Thread(target=fd.workflow_inference, args=(frame, (cap.get(3),cap.get(4),3),))
        # infer.daemon = True
        # infer.start()
        # (cap.get(3), cap.get(4), 3)输入图大小
        # fd.workflow_postprocess_detect(frame)
        ######
        # 打开数据库连接
        db = pymysql.connect(host='localhost', user='root', password='YLmm9600&', database='data_collect',
                             charset='utf8')

        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        #######
        for landmarks in fa.get_landmarks(frame, bboxes, calibrate=True):
            # 106个脸部特征点 len(landmarks)
            # calculate head pose
            _, euler_angle = hp.get_head_pose(landmarks)
            pitch, yaw, roll = euler_angle[:, 0]

            eye_markers = np.take(landmarks, fa.eye_bound, axis=0)

            eye_centers = np.average(eye_markers, axis=1)
            # eye_centers = landmarks[[34, 88]]

            # eye_lengths = np.linalg.norm(landmarks[[39, 93]] - landmarks[[35, 89]], axis=1)
            eye_lengths = (landmarks[[39, 93]] - landmarks[[35, 89]])[:, 0]

            iris_left = gs.get_mesh(frame, eye_lengths[0], eye_centers[0])
            pupil_left, _ = gs.draw_pupil(iris_left, frame, thickness=1)

            iris_right = gs.get_mesh(frame, eye_lengths[1], eye_centers[1])
            pupil_right, _ = gs.draw_pupil(iris_right, frame, thickness=1)

            pupils = np.array([pupil_left, pupil_right])

            poi = landmarks[[35, 89]], landmarks[[39, 93]], pupils, eye_centers
            theta, pha, delta = self.calculate_3d_gaze(frame, poi)

            if yaw > 30:
                end_mean = delta[0]
            elif yaw < -30:
                end_mean = delta[1]
            else:
                end_mean = np.average(delta, axis=0)

            if end_mean[0] < 0:
                zeta = arctan(end_mean[1] / end_mean[0]) + pi
            else:
                zeta = arctan(end_mean[1] / (end_mean[0] + 1e-7))

            # print(zeta * 180 / pi)
            # print(zeta)
            if roll < 0:
                roll += 180
            else:
                roll -= 180

            print("zete:", zeta)
            real_angle = zeta + roll * pi / 180
            # real_angle = zeta

            # print("end mean:", end_mean)
            # print(roll, real_angle * 180 / pi)

            R = norm(end_mean)
            offset = R * cos(real_angle), R * sin(real_angle)

            landmarks[[38, 92]] = landmarks[[34, 88]] = eye_centers

            # gs.draw_eye_markers(eye_markers, frame, thickness=1)
            print(theta, pha, delta)

            self.draw_sticker(frame, offset, pupils, landmarks)
            # 眼部凝视方向角
            cv2.putText(frame, "R_angle: " + "{:.2f}".format(real_angle), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 255, 0), thickness=2)  # GREEN
            # cv2.putText(frame, "Y: " + "{}".format(pha), (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            #         (255, 0, 0), thickness=2)  # BLUE
            # cv2.putText(frame, "Z: " + "{}".format(delta), (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            #        (0, 0, 255), thickness=2)  # RED
            # 头部三个角度：pitch, yaw, roll
            print(pitch, yaw, roll)
            ########
            # SQL 插入语句
            sql = "INSERT INTO data_collect.temp(r_angle, pitch, yaw, roll) VALUES(%s, %s, %s, %s)" % (
            real_angle, pitch, yaw, roll)
            try:
                # 执行sql语句
                cursor.execute(sql)
                # 执行sql语句
                db.commit()
            except:
                # 发生错误时回滚
                db.rollback()

                # 关闭数据库连接
            db.close()
            ########
        # 头部姿态欧拉角显示角度结果
        cv2.putText(frame, "pitch: " + "{:.2f}".format(euler_angle[0, 0]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), thickness=2)  # GREEN
        cv2.putText(frame, "yaw: " + "{:.2f}".format(euler_angle[1, 0]), (165, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), thickness=2)  # BLUE
        cv2.putText(frame, "roll: " + "{:.2f}".format(roll), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), thickness=2)  # RED
        frame = cv2.resize(frame, (960, 540))
        out.write(frame)
        # cv2.imshow('res', cv2.resize(frame, (960, 540)))
        # if cv2.waitKey(0) == ord('q'):
        #     break
        # os.system("E:\python\PyQtTest\dist\QtUi.exe")

        flag, self.image = frame

        show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage

    def load(self):
        conn = pymysql.Connect(
            host='localhost',
            user='root',
            password='YLmm9600&',
            database='data_collect',
        )
        cursor = conn.cursor()
        sql = "select * from temp"
        cursor.execute(sql)
        data = cursor.fetchall()
        row = len(data)
        y = 0
        for i in data[row - 1]:
            self.tableWidget.setItem(0, y, QtWidgets.QTableWidgetItem(str(data[row - 1][y])))
            y = y + 1
        cursor.close()
        conn.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = CamShow()
    ui.show()
    sys.exit(app.exec_())
