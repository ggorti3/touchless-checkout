import falcon
import falcon.asgi
from PIL import Image
import io
import torch
from torchvision import transforms
import asyncio
import aiofiles
import zxing
import json

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
    def __init__(model_path, img_in_size):
        self.model = mobilenetv2.get_model(
            num_classes=27,
            sample_size=112,
            width_mult=0.7)
        state_dict = torch.load("../results/jester_mobilenetv2_0.7x_RGB_16_best.pth", map_location=torch.device('cpu'))
        new_state_dict = {}
        for k, v in state_dict["state_dict"].items():
            k = k[7:]
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.CenterCrop(img_in_size),
            transforms.Resize(size=112)
        ])
    
    def save_img(self, pil_image):
        """
        Save an image to file so that we can call the barcode reader
        """
        pass
    
    def PIL_to_tensor(self, pil_images):
        """
        Convert PIL image to exact format for GesRec model
        """
        pass
    
    async def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200  # This is the default status
        resp.content_type = falcon.MEDIA_TEXT  # Default is JSON, so override
        resp.text = ('\nServer is running\n')
    
    async def on_post(self, req, resp):
        raw_data = json.load(req.bounded_stream)
        location_with_names = raw_data.get('location_with_names')

        data = await req.stream.read()
        image_id = str(self._config.uuid_generator())
        image = await self._store.save(image_id, data)

        resp.location = image.uri
        resp.media = image.serialize()
        resp.status = falcon.HTTP_201

# falcon.asgi.App instances are callable ASGI apps...
# in larger applications the app is created in a separate file
app = falcon.asgi.App(cors_enable=True)

# Resources are represented by long-lived class instances
healthcheck_res = HealthcheckResource()

# things will handle all requests to the '/things' URL path
app.add_route('/healthcheck', healthcheck_res)