# touchless-checkout
hackgt8 touchless checkout kiosk backend

This is the backend code for the Touchless Self-Checkout Kiosk. It provides a rest API so that the kiosk can query machine learning models that detect/classify hand gestures and item barcodes.

## Gesture Recognition

We used a pretrained deep learning model to detect and classify hand gestures in real time. The weights and model architecture files are open source:
```
@article{kopuklu_real-time_2019,
	title = {Real-time Hand Gesture Detection and Classification Using Convolutional Neural Networks},
	url = {http://arxiv.org/abs/1901.10323},
	author = {Köpüklü, Okan and Gunduz, Ahmet and Kose, Neslihan and Rigoll, Gerhard},
  year={2019}
}

@article{kopuklu2020online,
  title={Online Dynamic Hand Gesture Recognition Including Efficiency Analysis},
  author={K{\"o}p{\"u}kl{\"u}, Okan and Gunduz, Ahmet and Kose, Neslihan and Rigoll, Gerhard},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  volume={2},
  number={2},
  pages={85--97},
  year={2020},
  publisher={IEEE}
}

```

## Barcode Recognition

We used an open source image based barcode-reader library, zxing, to read strings from barcodes.

## REST API

The REST API has 3 endpoints: `/healthcheck`, `/process_images`, and `/barcode`:
- `/healthcheck`
    - get: send a response confirming status of the server
- `/process_images`
    - get: send a response confirming status of the Gesture Recognition Model
    - post: given 16 image frames in the request body, detect any hand gestures present and send results  back to kiosk
- `/barcode`
    - get: send a response confirming status of the barcode reader
    - post: given 1 image frame in the request body, detect any barcodes and their encoded string and send corresponding item info back to kiosk