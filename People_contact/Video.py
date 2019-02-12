import socket
import cv2
import pickle
import struct
import numpy as np

client =socket.socket(socket.AF_INET,socket.SOCK_STREAM) #socket
client.connect(('localhost',8080))
capture = cv2.VideoCapture(0) #내장 카메라 또는 외장 카메라에서 영상을 받아옵니다
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True: #영상 출력을 반복
    ret, frame = capture.read() #카메라의 상태 및 프레임을 받아옵니다
    """
    ret: 카메라의 상태가 저장되며 정상 작동할 경우 True를 반환
    frame : 현재 프레임이 저장
    """
    #cv2.imshow("VideoFrame", frame)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, frame2 = cv2.imencode('.jpg',frame,encode_param)
    data = np.array(frame2)
    str_data =data.tostring()
    size=len(str_data)
    print(size)
    client.send(str(len(str_data)).ljust(16).encode())
    client.send(str_data)
    cv2.imshow("VideoFrame", frame) #윈도우 창에 이미지를 띄웁니다
    if cv2.waitKey(1) > 0: break #키 입력이 있을 때 까지 while문을 반복

capture.release() #카메라 장치에서 받아온 메모리를 해제
cv2.destroyAllWindows() #모든 윈도우창을 닫습니다