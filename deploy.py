import torch
from models import mobilenetv2
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time
from collections import deque
import json
import io
import base64

# create custom dataloader

# model accepts 7 video frames at a time and performs a 3d convolution
# sample size param refers to dimensions of video input
# sample duration refers to length of time? for 7 frames?
# num samples for jester should be 1, in our app will be 1

# images are set to 112x112 and normalized
# jester images are 12 fps

# load model with pretrained weights

# profit?

def get_rev_label_dict(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    rev_label_dict = {}
    for line in lines:
        chunks = line.split(" ")
        label_int = int(chunks[0]) - 1
        label_str = chunks[1].strip()
        rev_label_dict[label_int] = label_str
    
    return rev_label_dict

def get_frames_from_jester_folder(path, general_transform):
    file_list = []
    for _, _, files in os.walk(path):
        for name in files:
            if name.startswith("00"):
                file_list.append(name)
    file_list.sort()

    frames_list = []
    for f in file_list:
        img_path = os.path.join(path, f)
        frame = Image.open(img_path)

        frame = general_transform(frame).to(torch.float)

        norm_transform = transforms.Normalize(
            mean=[torch.mean(frame[0]), torch.mean(frame[1]), torch.mean(frame[2])],
            #std=[torch.std(frame[0]), torch.std(frame[1]), torch.std(frame[2])]
            std=[1,1,1]
        )
        frame = norm_transform(frame)

        frames_list.append(frame)

    frames = torch.stack(frames_list)
    frames = torch.swapaxes(frames, 0, 1)
    return frames

def get_frames_from_mov(path, general_transform):
    cap = cv2.VideoCapture(path)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # Read until video is completed
    frames_list = []
    i = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if i % 2 == 0:
                frame = Image.fromarray(frame)


                frame = general_transform(frame).to(torch.float) / 255

                norm_transform = transforms.Normalize(
                    mean=[torch.mean(frame[0]), torch.mean(frame[1]), torch.mean(frame[2])],
                    std=[torch.std(frame[0]), torch.std(frame[1]), torch.std(frame[2])]
                    #std=[1,1,1]
                )

                frame = norm_transform(frame)

                frames_list.append(frame)


        # Break the loop
        else:
            break
        i += 1

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    return frames

def process_live_video(model, general_transform, rev_label_dict, spatial_length=16):
    softmax = nn.Softmax(dim=1)
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    else:
        print("Video stream opened successfully")


    # Read until video is completed
    frames_list = []
    i = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        _, _ = cap.read()
        ret, frame = cap.read()
        if ret == True:
            img = frame
            frame = Image.fromarray(frame)


            frame = general_transform(frame).to(torch.float)

            norm_transform = transforms.Normalize(
                mean=[torch.mean(frame[0]), torch.mean(frame[1]), torch.mean(frame[2])],
                #std=[torch.std(frame[0]), torch.std(frame[1]), torch.std(frame[2])]
                std=[1,1,1]
            )

            frame = norm_transform(frame)
            frames_list.append(frame)

            if i % spatial_length == spatial_length - 1:
                frames = torch.stack(frames_list)
                frames = torch.swapaxes(frames, 0, 1)
                out = model(frames.unsqueeze(0))
                scores = softmax(out)
                pred = torch.argmax(scores).item()
                conf = scores[0, pred].item()
                label_name = rev_label_dict[pred]

                info_str = print("{}: {}".format(label_name, conf))
                # ax.imshow(img)
                # ax.set_title(info_str)
                # plt.pause(0.05)
                # cv2.waitKey(1)

                frames_list = []
            


        # Break the loop
        else:
            break
        i += 1

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

def save_test_json_file(imgs_path, save_path):
    file_list = []
    for _, _, files in os.walk(imgs_path):
        for name in files:
            if name.startswith("00"):
                file_list.append(name)
    file_list.sort()
    file_list = file_list[12:28]

    frames_list = []
    transform = transforms.Resize(112)
    my_dict = {}
    for i, f in enumerate(file_list):
        img_path = os.path.join(imgs_path, f)
        frame = Image.open(img_path)
        frame = transform(frame)

        img_byte_arr = io.BytesIO()
        frame.save(img_byte_arr, format='jpeg')
        img_byte_arr = img_byte_arr.getvalue()
        encoded = base64.b64encode(img_byte_arr)

        my_dict["frame{}".format(i)] = encoded.decode('ascii')
    
    
    serialized_dict = json.dumps(my_dict)
    with open(save_path, "w") as f:
        f.write(serialized_dict)

if __name__ == "__main__":
    # model will run on jester dataset hopefully
    model = mobilenetv2.get_model(
        num_classes=27,
        sample_size=112,
        width_mult=0.7)

    state_dict = torch.load("../results/jester_mobilenetv2_0.7x_RGB_16_best.pth", map_location=torch.device('cpu'))
    new_state_dict = {}
    for k, v in state_dict["state_dict"].items():
        k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    softmax = nn.Softmax(dim=1)
    general_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.CenterCrop(size=720),
        transforms.Resize(size=112),
    ])

    # dummy = torch.rand((1, 3, 8, 112, 112))

    # out = model(dummy)
    # print(out)

    #frames = get_frames_from_jester_folder("../test_clips/jester_thumb_down/", general_transform)
    #frames = get_frames_from_mov("../test_clips/stop.mov", general_transform)

    # i = 22
    # frames = frames[:, i:16 + i]

    # plt.imshow(np.stack([frames[0, 0], frames[1, 0], frames[2, 0]], axis=2))
    # plt.show()
    # input()

    # start_time = time.time()
    # out = model(frames.unsqueeze(0))
    # print("--- %s seconds ---" % (time.time() - start_time))

    # print(softmax(out))

    # rev_label_dict = get_rev_label_dict("annotation_Jester/categories.txt")
    # process_live_video(model, general_transform, rev_label_dict, spatial_length=16)

    save_test_json_file("../test_clips/jester_thumb_down/", "../test_clips/sample_post_file.json")
    

