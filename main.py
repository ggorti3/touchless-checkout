import falcon
import falcon.asgi
import uvicorn

from resources import HealthcheckResource, ImageProcessResource, BarcodeResource


# falcon.asgi.App instances are callable ASGI apps...
# in larger applications the app is created in a separate file
app = falcon.asgi.App(cors_enable=True)

# Resources are represented by long-lived class instances
healthcheck_res = HealthcheckResource()
process_res = ImageProcessResource("weights/jester_mobilenetv2_0.7x_RGB_16_best.pth", "annotation_Jester/categories.txt")

catalog_dict = {}
catalog_dict["070972839564"] = ("notebook", "8.99")
catalog_dict["030000451304"] = ("energy_bars", "12.99")
catalog_dict["611269991000"] = ("energy_drink", "3.99")
catalog_dict["034000318285"] = ("candy_bar", "1.99")
barcode_res = BarcodeResource(catalog_dict)

# things will handle all requests to the '/things' URL path
app.add_route('/healthcheck', healthcheck_res)
app.add_route('/process_frames', process_res)
app.add_route('/barcode', barcode_res)

uvicorn.run(app, port=8000, log_level="debug")
