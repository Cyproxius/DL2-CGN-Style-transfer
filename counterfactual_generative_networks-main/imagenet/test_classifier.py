import imagenet.train_classifier as tc
import argparse

from dataloader.py import get_imagenet_dls

def main(args):

    model_path = args.model
    test_data = args.dataset

    # TODO do this more smartly (testing might take a long time)
    args.distributed = 0
    args.num_workers = 1
    args.multiprocessing_distributed = False

    # TODO Are we using this architecture?
    model_arch = 'resnet50'

    # Load in the model, pre-trained with above defined architecture
    model = InvariantEnsemble(model_arch, True)
    model.load_state_dict(torch.load(model_path))

    # Load in the dataset based on the selection of dataset
    if test_data = 'IN':
        dataloader = get_imagenet_dls(style_training="False", 
                                      imagenet_training="True",
                                      distributed=args.distributed, 
                                      batch_size=args.batch_size, 
                                      workers=args.num_workers)
    elif test_data = 'SIN':
        dataloader = get_imagenet_dls(style_training="True", 
                                      imagenet_training="False", 
                                      distributed=args.distributed, 
                                      batch_size=args.batch_size, 
                                      workers=args.num_workers)

    results = tc.validate(model, dataloader, cf_val_loader=None, dl_shape_bias=None, args=args)

    for k,v in results.items():
        print(f'{k}: {v}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="path to the model to test", default='') #TODO
    parser.add_argument("--dataset", type=str, help="what dataset to test on", choices=['IN', 'SIN'], default='IN')

    # Parse arguments blatantly copied from train_classifier.py
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')

    args = parser.parse_args()
    main(args)