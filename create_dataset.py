import os
import cv2
import glob
import random
import progressbar

import numpy as np

import matplotlib.pyplot as plt

rand_color = lambda : (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
rand_pos   = lambda a, b: (random.randint(a, b-1), random.randint(a, b-1))

target_size = 256
imgs_per_back = 30

backs = glob.glob('./dataset/backs/*.png')
fonts = glob.glob('./dataset/font_mask/*.png')

os.makedirs('./dataset/train/I', exist_ok=True)
os.makedirs('./dataset/train/Itegt', exist_ok=True)
os.makedirs('./dataset/train/Mm', exist_ok=True)
os.makedirs('./dataset/train/Msgt', exist_ok=True)

os.makedirs('./dataset/val/I', exist_ok=True)
os.makedirs('./dataset/val/Itegt', exist_ok=True)
os.makedirs('./dataset/val/Mm', exist_ok=True)
os.makedirs('./dataset/val/Msgt', exist_ok=True)

t_idx = len(os.listdir('./dataset/train/I'))
v_idx = len(os.listdir('./dataset/val/I'))

bar = progressbar.ProgressBar(maxval=len(backs)*imgs_per_back)
bar.start()
for back in backs:
    back_img = cv2.imread(back)
    bh, bw, _ = back_img.shape
    if bh < target_size or bw < target_size:
        back_img = cv2.resize(back_img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        bh, bw, _ = back_img.shape

    for bi in range(imgs_per_back):
        sx, sy = random.randint(0, bw-target_size), random.randint(0, bh-target_size)
        
        Itegt = back_img[sy:sy+target_size, sx:sx+target_size, :].copy()
        I     = Itegt.copy()
        Mm    = np.zeros_like(I)
        Msgt  = np.zeros_like(I)
        
        hist = []
        for font in random.sample(fonts, random.randint(2, 4)):
            font_img = cv2.imread(font)
            mask_img = np.ones_like(font_img, dtype=np.uint8)*255
            
            height, width, _ = font_img.shape
            
            angle = random.randint(-30, +30)
            fs = random.randint(90, 120)
            ratio = fs / height - 0.2
            
            matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, ratio)
            font_rot = cv2.warpAffine(font_img, matrix, (width, height), cv2.INTER_CUBIC)
            mask_rot = cv2.warpAffine(mask_img, matrix, (width, height), cv2.INTER_CUBIC)
            
            h, w, _ = font_rot.shape
            
            font_in_I = np.zeros_like(I)
            mask_in_I = np.zeros_like(I)
            
            allow = 0
            while True:
                sx, sy = rand_pos(0, target_size-w)
                
                done = True
                for sx_, sy_ in hist:
                    if (sx_ - sx)**2 + (sy_ - sy)**2 < (fs * ratio)**2 - allow:
                        done = False
                        break
                allow += 5
                
                if done:
                    hist.append([sx, sy])
                    break
            
            font_in_I[sy:sy+h, sx:sx+w, :] = font_rot
            mask_in_I[sy:sy+h, sx:sx+w, :] = mask_rot
            
            font_in_I[font_in_I > 30] = 255
            mask_in_I[mask_in_I > 30] = 255
            
            I = cv2.bitwise_and(I, 255-font_in_I)
            I = cv2.bitwise_or(I, (font_in_I // 255 * rand_color()).astype(np.uint8))
            
            Mm = cv2.bitwise_or(Mm, mask_in_I)
            Msgt = cv2.bitwise_or(Msgt, font_in_I)
        
        if bi < imgs_per_back*0.8:
            cv2.imwrite(f'dataset/train/I/{t_idx}.png', I)
            cv2.imwrite(f'dataset/train/Itegt/{t_idx}.png', Itegt)
            cv2.imwrite(f'dataset/train/Mm/{t_idx}.png', Mm)
            cv2.imwrite(f'dataset/train/Msgt/{t_idx}.png', Msgt)
            t_idx += 1
        else:
            cv2.imwrite(f'dataset/val/I/{v_idx}.png', I)
            cv2.imwrite(f'dataset/val/Itegt/{v_idx}.png', Itegt)
            cv2.imwrite(f'dataset/val/Mm/{v_idx}.png', Mm)
            cv2.imwrite(f'dataset/val/Msgt/{v_idx}.png', Msgt)
            v_idx += 1
            
        bar.update(t_idx + v_idx)
bar.finish()