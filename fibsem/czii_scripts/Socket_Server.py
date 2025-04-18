import socket
import threading

import numpy as np
import io
import matplotlib.pyplot as plt
import pickle
import os
from Odemis_Control_Functions import *

odemis = OdemisControl()

def insert_objective(conn, args):
    try:
        odemis.insert_objective()
    except Exception as e:
        print(f"failed because of {e}")

def send_tiff(conn, args):
    if not args:
        print("No path provided.")
        return
    path = " ".join(args)
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        conn.sendall((0).to_bytes(8, 'big'))
        return
    try:
        image_dict = odemis.load_tif_as_array(path)
        data = pickle.dumps(image_dict)
        conn.sendall(len(data).to_bytes(8, 'big'))
        conn.sendall(data)
        print("Sent TIFF image and metadata.")
    except Exception as e:
        print(f"An error occured sending the file: {e}")

def greet(conn, args):
    print("Hello from the server!")

def add(conn, args):
    print(f"The sum is {int(args[0]) + int(args[1])}.")


FUNCTIONS = {
    "greet": greet,
    "add": add,
    "insert_objective": insert_objective,
    "send_tiff": send_tiff
}

def handle_client(conn):
    with conn:
        data = conn.recv(1024).decode().strip()
        if not data:
            return
        print(f"Received: {data}")
        parts = data.split()
        cmd, args = parts[0], parts[1:]
        if cmd in FUNCTIONS:
            FUNCTIONS[cmd](conn, args)
        else:
            print("Unknown command.")


def main():
    HOST = '0.0.0.0'
    PORT = 23924
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        while True:
            conn, _ = s.accept()
            threading.Thread(target=handle_client, args=(conn,)).start()

print('Test is running.')
main()