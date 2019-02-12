import socket
import cv2
import struct
import numpy as np
import pickle

ip='localhost'
port = 8080
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM) #socket
server.bind((ip,port)) #bind : connect with client
server.listen(20) #client 연결요청을 받기 위한 method
payload_size = struct.calcsize(">L")

def handler(client):
    while True:
        length = _recvall(client, 16)
        str_data = _recvall(client, int(length))
        print(len(str_data))
        data = np.fromstring(str_data, dtype='uint8')
        img = cv2.imdecode(data)
        cv2.imshow('ImageWindow', img)


def _recvall(sock,count):
    buf=b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf +=newbuf
        count -=len(newbuf)
        return buf

while True:
    client ,addr = server.accept()
    print(addr)
    handler(client)