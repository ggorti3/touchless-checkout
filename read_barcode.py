import zxing
from PIL import Image
import time

def read_barcode(reader, img_path):
    barcode = reader.decode(img_path, try_harder=True)
    return barcode.parsed

if __name__ == "__main__":
    reader = zxing.BarCodeReader()
    start_time = time.time()
    result = read_barcode(reader, "../test_clips/IMG_4171.JPG")
    print("--- %s seconds ---" % (time.time() - start_time))

    print(result)