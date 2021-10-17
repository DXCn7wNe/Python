from PIL import Image, ImageDraw, ImageFont

input_image_name = 'a.jpg'
font_path = "/usr/share/fonts/truetype/migmix/migu-1c-regular.ttf"
image_dpi = 100
space_ratio = 0.1
tick_length = 10
scale_ratio = 0.2

image = Image.open(input_image_name)
draw = ImageDraw.Draw(image)
draw.font = ImageFont.truetype(font_path, 48)
# draw.line((100,200,300,400), fill=(255,0,0),width=2)
(space_left,space_down) = (int(float(image.size[0]) * space_ratio), int(float(image.size[1]) * space_ratio))
scale_length = int(float(image.size[0]) * scale_ratio)

draw.line((space_left+0, image.size[1]-space_down+0, space_left+scale_length, image.size[1]-space_down), fill=(255,0,0),width=2)
draw.line((space_left, image.size[1]-space_down-tick_length/2, space_left, image.size[1]-space_down+tick_length/2), fill=(255,0,0),width=2)
draw.line((space_left+scale_length, image.size[1]-space_down-tick_length/2, space_left+scale_length, image.size[1]-space_down+tick_length/2), fill=(255,0,0),width=2)
draw.text((space_left+scale_length/2, image.size[1]-space_down-tick_length), "30um", (255,0,0), anchor='mm')
image.show()


