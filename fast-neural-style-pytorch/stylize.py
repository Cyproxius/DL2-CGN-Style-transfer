import torch
import utils
import transformer
import os
from torchvision import transforms
import time
import cv2
import numpy as np
import csv
import itertools

STYLE_TRANSFORM_PATH = "transforms/wave.pth"
PRESERVE_COLOR = False


def stylize_imagenet(style_path, content_folder, save_folder):
    """
    Reads frames/pictures as follows:

    content_folder
        pic1.ext
        pic2.ext
        pic3.ext
        ...

    and saves as the styled images in save_folder as follow:

    save_folder
        pic1.ext
        pic2.ext
        pic3.ext
        ...
    """
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Load Transformer Network
    net = transformer.TransformerNetwork()
    net.load_state_dict(torch.load(style_path))
    net = net.to(device)

    # Stylize every frame
    images = np.array([img for img in os.listdir(content_folder)])

    with torch.no_grad():
        for image_name in images:
            t = time.time()
            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()
            # Load content image
            content_image = utils.load_image(content_folder + image_name)
            content_tensor = utils.itot(content_image).to(device)

            # Generate image
            generated_tensor = net(content_tensor)
            generated_image = utils.ttoi(generated_tensor.detach())
            if (PRESERVE_COLOR):
                generated_image = utils.transfer_color(content_image, generated_image)
            # Save image
            out_img_name = image_name.split('.')[0] + "_style_" + style_path.split('/')[-1].split('.')[0] + '.JPEG'
            utils.saveimg(generated_image, save_folder + out_img_name)


def stylize_cgn(style_path, content_folder, save_folder):
    """
    Reads frames/pictures as follows:

    content_folder
        pic1.ext
        pic2.ext
        pic3.ext
        ...

    and saves as the styled images in save_folder as follow:

    save_folder
        pic1.ext
        pic2.ext
        pic3.ext
        ...
    """
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Load Transformer Network
    net = transformer.TransformerNetwork()
    net.load_state_dict(torch.load(style_path))
    net = net.to(device)

    out_labels_fn = '/'.join(save_folder.split('/')[:-2])+'/labels.csv'

    if os.path.isfile(out_labels_fn):
        labels = out_labels_fn
    else:
        labels = '/'.join(content_folder.split('/')[:-2])+'/labels.csv'
    
    # Read in the raw labels csv file
    with open(labels) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        rows = [row for row in csv_reader]
    header, label_rows = rows[0], rows[1:]

    # Get the 'base name' of all labels (this includes duplicates)
    label_rows_unique = [[row[0].split("_style_")[0], *row[1:]] for row in label_rows]

    # Remove the duplicates
    label_rows_unique.sort()
    label_rows_unique = list(label_rows_unique for label_rows_unique,_ in itertools.groupby(label_rows_unique))

    # Create the new labels for the images with the current style
    new_labels = [[row[0] + "_style_" + style_path.split('/')[-1].split('.')[0], *row[1:]] for row in label_rows_unique]
    all_new_labels = label_rows + new_labels
    all_new_labels.sort()

    # Combine the new labels with the old labels
    all_new_rows = header + all_new_labels

    # Overwrite the old labels file
    with open(out_labels_fn, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(all_new_rows)

    images = np.array([img for img in os.listdir(content_folder)])

    with torch.no_grad():
        for image_name in images:
            t = time.time()
            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()
            # Load content image
            content_image = utils.load_image(content_folder + image_name)
            content_tensor = utils.itot(content_image).to(device)

            # Generate image
            generated_tensor = net(content_tensor)
            generated_image = utils.ttoi(generated_tensor.detach())
            if (PRESERVE_COLOR):
                generated_image = utils.transfer_color(content_image, generated_image)
            # Save image
            label_img_name = image_name.split('.')[0].split("_x_gen")[0]
            out_img_name = label_img_name + "_style_" + style_path.split('/')[-1].split('.')[0] + '_x_gen.JPEG'
            utils.saveimg(generated_image, save_folder + out_img_name)