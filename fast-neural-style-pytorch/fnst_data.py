from stylize import *
import argparse

def main(args):

    styles = ['transforms/starry.pth','transforms/wave.pth','transforms/mosaic.pth','transforms/lazy.pth']

    if args['dataset'] == 'imagenet':
        for t in ['train', 'val']:
            for style_path in styles:
                class_dirs = os.listdir(args['content_folder']+f'/{t}')
                for class_img in class_dirs:
                    source_folder = args['content_folder']+f'/{t}/'+class_img+'/'
                    dest_folder = args['destination_folder']+f'/{t}/'+class_img+'/'

                    if not os.path.isdir(dest_folder):
                        os.mkdir(dest_folder)

                    stylize_imagenet(style_path, source_folder, dest_folder)

    elif args['dataset'] == 'cgn':
        for t in ['train', 'val']:
            for style_path in styles:
                source_folder = args['content_folder']+f'/{t}/ims/'
                dest_folder = args['destination_folder']+f'/{t}/ims/'

                stylize_cgn(style_path, source_folder, dest_folder)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    """
    This code is hardcoded for the INmini dataset and CGN-INmini datasets.

    The code assumes the structure of the INmini dataset to be:
    /dataset_folder
        /INmini
            /train
                /img_class_dir_1
                    img1.ext
                    img2.ext
                    etc
                /img_class_dir_2
                etc
            /val
                /img_class_dir_1
                    img1.ext
                    img2.ext
                    etc
                /img_class_dir_2
                etc
    The generated SINmini will be saved in the same structure.

    And the structure of the CGN-INmini dataset is assumed to be:
    /dataset_folder
        /CGN-INmini
            /train
                /ims
                    img1.ext
                    img2.ext
                    etc
                labels.csv
            /val
                /ims
                    img1.ext
                    img2.ext
                    etc
                labels.csv
    """
    parser.add_argument("--dataset", type=str, help="what dataset to apply the style transfer on", choices=['cgn', 'imagenet'], default='imagenet')
    # Default content folder for:
        # Imagenet: ../../datasets/INmini
        # CGN: ../../datasets/CGN-INmini
    parser.add_argument("--content_folder", type=str, help="image content folder", default='../../datasets/INmini')
    # Default destination folder for:
        # Imagenet: ../../datasets/SINmini
        # CGN: ../../datasets/CGN-SINmini
    
    """NOTE PLEASE READ NOTE"""
    # these destination folders must have (empty) /train/ims and /val/ims folders
    # defined, otherwise the script won't run  
    parser.add_argument("--destination_folder", type=str, help="folder where to save styled images", default='../../datasets/SINmini')

    args = parser.parse_args()
    main(vars(args))