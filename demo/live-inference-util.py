import serial
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib qt
import struct
import imgConverter
import cv2
from PIL import Image

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=30)  # open serial port
print(f"Succesfully opened: {ser.name}")


while(1):
    # Flushing the buffer
    ser.read_all()

    buffer = bytearray("", "utf8")
    tocken = bytearray("New image", "utf8")
    # Continously reading until start message received

    while tocken not in buffer:
            buffer = ser.readline()
    print("Buffer synchronised!")
    # Now we are synchronised!
    # We know the content of the payload, so we read and parse it

    # There are 5 bytes encoding the image dimension (2 byte each dimension)
    # And one byte encoding the lenght of the pixel formats
    img_resol = ser.read(4) # timeout
    # We are now dealing with bytes
    # Which are not encoded to make sense to humans
    # In this case we know that in the first two bytes there is an integer,
    # and they are big ending. So we proceed to decode it!
    x_size = int.from_bytes(img_resol[0:2], byteorder='big')
    y_size = int.from_bytes(img_resol[2:4], byteorder='big')
     # Now receive the boundary box and face probability
    bbox = ser.read(10)

    x = int.from_bytes(bbox[0:2], byteorder='big')
    y = int.from_bytes(bbox[2:4], byteorder='big')
    w = int.from_bytes(bbox[4:6], byteorder='big')
    h = int.from_bytes(bbox[6:8], byteorder='big')
    prob = int.from_bytes(bbox[8:10], byteorder='big')
    
    print("Face detected at: x: ",x,",y: ",y,", w: ",w,", h: ",h,", prob.: ",prob)
    # Receive the length of the image format descriptor
    img_resol = ser.read(1)
    pixel_format_len = int.from_bytes(img_resol, byteorder='big')
    print(f"X dimension: {x_size}, Y dimension: {y_size}, pixel format length: {pixel_format_len}")
    
    # And now we receive the pixel format
    # Which is encoded in a string
    img_format = ser.read(pixel_format_len) # timeout
    print("Image format: ", img_format.decode("utf-8"))

    # Now we expect 4 bytes encoding the lenght of the image
    img_dim = ser.read(4)
    img_dim = int.from_bytes(img_dim, byteorder='big')
    # We are using the RGB888, so 24 bits per pixel
    # At a resolution of 48*48 pixels, this is 6912 bytes
    print("Total image dimension: ", img_dim)

    # Now we can finally acquire the image...
    img = ser.read(img_dim)

    # Every 3 bytes there is a pixel
    img = parse_image_rgb888(img, x_size, y_size)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.imshow(img/255)
    # imgConverter.convert(img, "live_inference.png", x_size, y_size, img_format.decode('ASCII'))

    # image_saved = Image.open("live_inference.png")

    # Display the image
    # ax.imshow(image_saved)
    # Add the bboxes and probability
    if prob > 40:
        # Plot the bbox
        ax.plot([x, x+w], [y, y], color='red', linewidth=2)  # Top line
        ax.plot([x, x], [y, y+h], color='red', linewidth=2)  # Left line
        ax.plot([x+w, x+w], [y, y+h], color='red', linewidth=2)  # Right line
        ax.plot([x, x+w], [y+h, y+h], color='red', linewidth=2)  # Bottom line
        ax.text(x+2, y+2, f'Face: {prob:d}%', color='red', fontsize=12, bbox=dict(facecolor='white', edgecolor='red'))
        print("Face detected!")
    else:
        # No bbox if low probability
        ax.text(10, 5, f'Face: {(prob):.0f}%', color='red', fontsize=12, bbox=dict(facecolor='white', edgecolor='red'))
        print("No face detected!")
    plt.xlim(0, x_size)
    plt.ylim(y_size, 0)
    plt.show()
