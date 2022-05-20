from stylize import *
import argparse

def main(args):

    style = args['style']

    if style == 'starry':
        style_path = 'transforms/starry.pth'
    elif style == 'waves':
        style_path = 'transforms/wave.pth'
    elif style == 'mosaic':
        style_path = 'transforms/mosaic.pth'
    elif style == 'lazy':
        style_path = 'transforms/lazy.pth'

    class_dirs = os.listdir(args['content_folder'])
    for class_img in class_dirs:
        source_folder = args['content_folder'] + '/' + class_img + '/'
        dest_folder = args['destination_folder'] + '/' + class_img + '/'

        if not os.path.isdir(dest_folder):
            os.mkdir(dest_folder)

        stylize_folder_single(style_path, source_folder, dest_folder, args['ratio'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    """
    This code structure only works for images saved in the format:
    
    folder_containing_all_imgs
        imgs_class_1
            img1.ext
            img2.ext
            etc
        img_class_2
            img3.ext
            img4.ext
        etc
    """
    parser.add_argument("--content_folder", type=str, help="image content folder", default='data/content_imgs')
    parser.add_argument("--destination_folder", type=str, help="folder where to save styled images", default='data/result_imgs')
    parser.add_argument("--ratio", type=float, help="fraction of dataset to transform", default=1)
    parser.add_argument("--style", type=str, help="style transformation to apply", choices=['starry', 'waves', 'mosaic', 'lazy'], default='starry')

    args = parser.parse_args()
    main(vars(args))
    #NOTE: Don't forget to specify if you want to transform the train or val folder