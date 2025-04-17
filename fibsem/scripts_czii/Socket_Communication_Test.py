import socket
import matplotlib.pyplot as plt

SERVER_IP = '10.50.2.119'
PORT = 23924

path = '/mnt/meteor_shared/AutoLamella/AutoLamella-2025-03-26-17-58-DEV-TEST/02-fair-mouse/ref_alignment_ib.tif'
command = f"send_tiff {path}"

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((SERVER_IP, PORT))
    s.sendall(command.encode("utf-8"))

    size = int.from_bytes(s.recv(8), 'big')
    data = b""
    while len(data) < size:
        chunk = s.recv(min(4096, size - len(data)))
        if not chunk:
            break
        data += chunk

    import pickle
    bundle = pickle.loads(data)
    image = bundle['image']
    metadata = bundle['metadata']
    plt.imshow(image)
    plt.show()
    print(f"The received metadata are{metadata}.")

