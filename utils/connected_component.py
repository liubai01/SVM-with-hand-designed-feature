import pickle
import numpy as np
import matplotlib.pyplot as plt

class Pixel():

    def __init__(self, w, h, is_blank):
        self.w = w
        self.h = h
        self.is_blank = is_blank
        self.has_traversed = False
        self.near_pts = []
        self.after_pixel = None
        self.before_pixel = None
        self.id = 0

    def detach(self):
        if self.before_pixel:
            self.before_pixel.after_pixel = self.after_pixel
        if self.after_pixel:
            self.after_pixel.before_pixel = self.before_pixel


def get_index(w, h, width=28):
    return width * w + h

def get_components(x):
    x_thre = x < 1e-3
    width = x.shape[1]
    height = x.shape[0]
    # generate pixels
    header = Pixel(None, None, None)
    now_p = header
    pixels = []
    for w in range(width):
        for h in range(height):
            new_p = Pixel(w, h, x_thre[h, w])
            pixels.append(new_p)
            now_p.after_pixel = new_p
            new_p.before_pixel = now_p
            now_p = new_p

    # connect them
    for w in range(width):
        for h in range(height):
            i = get_index(w,h)
            p = pixels[i]
            if p.is_blank:
                if w != 0:
                    obj_p = pixels[get_index(w - 1, h)]
                    if obj_p.is_blank:
                        p.near_pts.append(obj_p)
                if w != width - 1:
                    obj_p = pixels[get_index(w + 1, h)]
                    if obj_p.is_blank:
                        p.near_pts.append(obj_p)
                if h != 0:
                    obj_p = pixels[get_index(w, h - 1)]
                    if obj_p.is_blank:
                        p.near_pts.append(obj_p)
                if h != height - 1:
                    obj_p = pixels[get_index(w, h + 1)]
                    if obj_p.is_blank:
                        p.near_pts.append(obj_p)

    comps = []
    comps_id = 1
    while header.after_pixel is not None:
        if header.after_pixel.is_blank:
            comp_list = [header.after_pixel]
            TODO_list = [header.after_pixel]
            while len(TODO_list) > 0:
                p = TODO_list[0]
                for to_p in p.near_pts:
                    if to_p.has_traversed == False:
                        to_p.has_traversed = True
                        TODO_list.append(to_p)
                        comp_list.append(to_p)
                TODO_list.pop(0)
            comps.append(len(comp_list))
            for p in comp_list:
                p.id = comps_id
                p.detach()
            comps_id += 1
        else:
            header.after_pixel.detach()
    comps = np.array(comps)
    if len(comps) >= 2:
        second_maximum = comps[-2]
    else:
        second_maximum = comps[-1]
    return np.count_nonzero(comps > 10), second_maximum