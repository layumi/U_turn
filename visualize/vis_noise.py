from  PIL import Image
import numpy

image_path = "0001/0001_c1s1_001051_00.jpg"

original_im = Image.open("/home/zzd/Market/pytorch/query/" + image_path)
original_im = original_im.resize((128,256))

attack_im = Image.open("../attack_query/pytorch/query/" + image_path)

diff = numpy.array(original_im, dtype=float) - numpy.array(attack_im, dtype=float)

# move to 128 for show
diff += 128
diff = Image.fromarray( numpy.uint8(diff))

im_save = Image.new('RGB',(128*3, 256))
im_save.paste( original_im, (0,0))
im_save.paste( diff, (128,0))
im_save.paste( attack_im, (256,0))
im_save.save('vis_noise.jpg')
