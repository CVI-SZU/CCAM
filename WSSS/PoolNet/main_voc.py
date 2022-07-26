import argparse
import os
from dataset.dataset import get_loader_voc
from solver import Solver

def main(config):
    if config.mode == 'train':
        train_loader = get_loader_voc(config)
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_folder, run)):
            run += 1
        os.mkdir("%s/run-%d" % (config.save_folder, run))
        os.mkdir("%s/run-%d/models" % (config.save_folder, run))
        config.save_folder = "%s/run-%d" % (config.save_folder, run)
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        config.test_root = config.train_root
        test_loader = get_loader_voc(config, mode='test')
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")

if __name__ == '__main__':

    vgg_path = './dataset/pretrained/vgg16_20M.pth'
    resnet_path = './dataset/pretrained/resnet50_caffe.pth'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5) # Learning rate resnet:5e-5, vgg:1e-4
    parser.add_argument('--wd', type=float, default=0.0005) # Weight decay
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')

    # Training settings
    parser.add_argument('--arch', type=str, default='resnet') # resnet or vgg
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=9)
    parser.add_argument('--batch_size', type=int, default=1) # only support 1 now
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='./results')
    parser.add_argument('--sal_folder', type=str, default='./results/refined_background_cues')

    parser.add_argument('--epoch_save', type=int, default=3)
    parser.add_argument('--iter_size', type=int, default=10)
    parser.add_argument('--show_every', type=int, default=50)

    # Train data
    parser.add_argument('--train_root', type=str, default='')
    parser.add_argument('--pseudo_root', type=str, default='')
    parser.add_argument('--train_list', type=str, default='')

    # Testing settings
    parser.add_argument('--model', type=str, default='results/run-55/models/epoch_9.pth') # moco Snapshot
    # parser.add_argument('--model', type=str, default='poolnet-final/final.pth') # moco Snapshot

    parser.add_argument('--test_fold', type=str, default='visual_results') # Test results saving folder
    parser.add_argument('--sal_mode', type=str, default='v') # Test image dataset

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()

    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)
    if not os.path.exists(config.sal_folder):
        os.mkdir(config.sal_folder)
    if not os.path.exists(config.test_fold):
        os.mkdir(config.test_fold)

    # Get test set info
    # test_root, test_list = get_test_info(config.sal_mode)
    # config.test_root = test_root
    # config.test_list = test_list

    main(config)
