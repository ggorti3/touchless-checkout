import falcon
import falcon.asgi
from PIL import Image
import io
import torch
import numpy as np
from torchvision import transforms
import asyncio
import aiofiles
import zxing
import json
from deploy import get_rev_label_dict
from models import mobilenetv2
import logging
import base64

logging.basicConfig(level=logging.DEBUG)   # add this line
logger = logging.getLogger("foo")

# Falcon follows the REST architectural style, meaning (among
# other things) that you think in terms of resources and state
# transitions, which map to HTTP verbs.
class HealthcheckResource:
    async def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200  # This is the default status
        resp.content_type = falcon.MEDIA_TEXT  # Default is JSON, so override
        resp.text = ('\nServer is running\n')

class ImageProcessResource:
    def __init__(self, model_path, classes_path):
        self.model = mobilenetv2.get_model(
            num_classes=27,
            sample_size=112,
            width_mult=0.7)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        new_state_dict = {}
        for k, v in state_dict["state_dict"].items():
            k = k[7:]
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        self.softmax = torch.nn.Softmax(dim=1)
        self.rev_label_dict = get_rev_label_dict(classes_path)

        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.CenterCrop(112),
            # transforms.Resize(size=112)
        ])
    
    def save_img(self, pil_image):
        """
        Save an image to file so that we can call the barcode reader
        """
        pass
    
    def PIL_to_tensor(self, bytes_list):
        """
        Convert PIL image to exact format for GesRec model

        returns a tensor for the model
        """
        tensor_list = []
        for frame_bytes in bytes_list:
            frame = Image.open(io.BytesIO(base64.b64decode(frame_bytes)))

            frame = self.transform(frame).to(torch.float)
            norm_transform = transforms.Normalize(
                mean=[torch.mean(frame[0]), torch.mean(frame[1]), torch.mean(frame[2])],
                std=[1,1,1]
            )
            frame = norm_transform(frame)
            tensor_list.append(frame)
        x = torch.stack(tensor_list)
        x = torch.swapaxes(x, 0, 1).unsqueeze(0)
        return x
    
    def process_scores(self, scores):
        """
        Given a tensor of classification scores, return serialized json dictionary containing top 5 classes and confidences
        """
        scores = scores.detach().cpu().numpy().flatten()
        top_five_idx = np.argsort(scores)[::-1][:5]
        my_dict = {}
        for i, idx in enumerate(top_five_idx):
            idx = idx.item()
            my_dict[i] = (self.rev_label_dict[idx], scores[idx].item())
        return json.dumps(my_dict)
        
    
    async def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200  # This is the default status
        resp.content_type = falcon.MEDIA_TEXT  # Default is JSON, so override
        resp.text = ('\nProcess frames resource is running\n')
    
    async def on_post(self, req, resp):
        logger.debug('This is a debug message')
        loop = asyncio.get_running_loop()

        obj = await req.get_media()

        # raw_data = await loop.run_in_executor(None, json.load(), req.bounded_stream)
        bytes_list = []
        for i in range(0, 16):
            frame_bytes = obj.get("frame{}".format(i))
            bytes_list.append(frame_bytes)
        
        x = await loop.run_in_executor(None, self.PIL_to_tensor, bytes_list)
        logger.debug("frames converted to tensor")
        out = await loop.run_in_executor(None, self.model, x)
        logger.debug("output generated from gesrec model")
        scores = await loop.run_in_executor(None, self.softmax, out)
        logger.debug("scores generated from output")
        resp.media = await loop.run_in_executor(None, self.process_scores, scores)
        logger.debug("info returned")

# falcon.asgi.App instances are callable ASGI apps...
# in larger applications the app is created in a separate file
app = falcon.asgi.App(cors_enable=True)

# Resources are represented by long-lived class instances
healthcheck_res = HealthcheckResource()
process_res = ImageProcessResource("../results/jester_mobilenetv2_0.7x_RGB_16_best.pth", "annotation_Jester/categories.txt")

# things will handle all requests to the '/things' URL path
app.add_route('/healthcheck', healthcheck_res)
app.add_route('/process_frames', process_res)


import uvicorn
uvicorn.run(app, port=8000, log_level="debug")