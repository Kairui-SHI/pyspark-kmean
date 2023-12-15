from PIL import Image

def bmp2tif(filepath):
    image = Image.open(filepath).convert('RGB')
    image.save('output.tif', format='TIFF', compression='tiff_lzw')
    image.close()
    return True

def tif_rgb2gray():
    rgb_image = Image.open('output.tif')
    image = gray_image = rgb_image.convert('L')
    gray_image.save('output_gray.tif')
    rgb_image.close()
    return image

def get_pixelvalue(image):
    pixels = list(image.getdata())
    with open('pixel_values.txt', 'w') as file:
        for pixel in pixels:
            # 写入像素值到文本文件
            file.write(f'{pixel}\n')

def produce_img(pixels,width,height):
    new_image = Image.new("L", (width, height))
    new_image.putdata(pixels)
    new_image.save('classify.png')
    return new_image