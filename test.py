

#for y in range(0, 100-5, 5):
#    print(f"y {y}")


im_height = 20
im_width = 20
patch_size = 6

    
height_divs = im_height // patch_size
width_div = im_width // patch_size


for y in range(height_divs):
    for x in range(width_div):
                print(f"X:{x}, Y:{y}")
                ## Store x, y cordinates in some way?