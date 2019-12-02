import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import helpers as hp

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img): #initial image is float values
    rimg = img - np.min(img) #min of each color
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8) # converts the pixel colors to scale 0 to 255
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8) #initializes a new gt image, the last param is of length 3
        gt_img8 = img_float_to_uint8(gt_img)  #converts float values to rgb scale        
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h] #covert it into patches
            else:
                im_patch = im[j:j+w, i:i+h, :] #covert into patches and keep the extra data
            list_patches.append(im_patch)
    return list_patches

def extract_features_rgb(img):
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat = np.append(feat_m, feat_v)
    return feat

def extract_features_aug(img):
    r=np.mean(img[:,:,0])
    g=np.mean(img[:,:,1])
    b=np.mean(img[:,:,2])
    r_var=np.var(img[:,:,0])
    g_var=np.var(img[:,:,1])
    b_var=np.var(img[:,:,2])
    rg_s=np.mean(img[:,:,0]+img[:,:,1])/2
    rb_s=np.mean(img[:,:,0]+img[:,:,2])/2
    gb_s=np.mean(img[:,:,1]+img[:,:,2])/2
    rgb_s=np.mean((img[:,:,0]+img[:,:,1]+img[:,:,2]))/3
    rg_var_s = np.var(((img[:,:,0])+(img[:,:,1]))/2)
    rb_var_s = np.var(((img[:,:,0])+(img[:,:,2]))/2)
    gb_var_s = np.var(((img[:,:,1])+(img[:,:,2]))/2)
    rgb_var_s = np.var(img)
    rg=np.mean(img[:,:,0]*img[:,:,1])
    rb=np.mean(img[:,:,0]*img[:,:,2])
    gb=np.mean(img[:,:,1]*img[:,:,2])
    rgb=np.mean(img[:,:,0]*img[:,:,1]*img[:,:,2])
    rg_var=np.var(img[:,:,0]*img[:,:,1])
    rb_var=np.var(img[:,:,0]*img[:,:,2])
    gb_var=np.var(img[:,:,1]*img[:,:,2])
    rgb_var=np.var(img[:,:,0]*img[:,:,1]*img[:,:,2])
    return r,g,b,r_var,g_var,b_var,rg_s,rb_s,gb_s,rgb_s,rg_var_s,rb_var_s,gb_var_s,rgb_var_s,rg,rb,gb,rgb,rg_var,rb_var,gb_var,rgb_var

# Extract features for a given image
def extract_img_features(filename):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    return X

def extract_img_features_rgb(filename):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([ extract_features_rgb(img_patches[i]) for i in range(len(img_patches))])
    return X

def extract_img_gt(filename,patch_size):
    gt=load_image(filename)
    gt_patches=img_crop(gt, patch_size, patch_size)
    Y= np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])
    
def value_to_class(v,foreground_threshold):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img