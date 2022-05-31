"""
ref. from https://github.com/zxhuang1698/interpretability-by-parts
ref. from https://github.com/Sierkinhane/ORNet
modified by sierkinhane
"""
import argparse
import time
import torch.backends.cudnn as cudnn
from utils import *
from dataset.cub200 import *
from models.loss import *
from models.model import *
from torchvision import transforms
import yaml
from easydict import EasyDict as edict
import random
import logging
import pprint

# benchmark before running
cudnn.benchmark = True
os.environ["NUMEXPR_NUM_THREADS"] = "16"
flag = True

def parse_arg():
    parser = argparse.ArgumentParser(description="train CCAM on CUB dataset")
    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default='config/CCAM_CUB.yaml')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--experiment', type=str, required=True, help='record different experiments')
    parser.add_argument('--pretrained', type=str, required=True, help='adopt different pretrained parameters, [supervised, mocov2, detco]')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f)
        config = edict(config)
    config.EXPERIMENT = args.experiment
    config.LR = args.lr
    config.BATCH_SIZE = args.batch_size
    config.PRETRAINED = args.pretrained

    return config, args


def main():
    config, args = parse_arg()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    logger.info(pprint.pformat(config))

    if config.SEED != -1:
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)

    print("=> creating log folder...")
    creat_folder(config, args)

    # log
    sys.stdout = Logger('{}/{}_log.txt'.format(config.LOG_DIR, config.EXPERIMENT))

    # create model
    print("=> creating model...")
    model = get_model(pretrained=config.PRETRAINED).cuda()
    param_groups = model.get_parameter_groups()
    # model_info(model)

    criterion = [SimMaxLoss(metric='cos', alpha=args.alpha).cuda(), SimMinLoss(metric='cos').cuda(),
                 SimMaxLoss(metric='cos', alpha=args.alpha).cuda()]

    # data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # we follow PSOL to adopt 448x448 as input to generate pseudo bounding boxes
    test_transforms = transforms.Compose([
        transforms.Resize(size=(480, 480)),
        transforms.CenterCrop(size=(448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # wrap to dataset
    train_data = CUB200(root=config.ROOT, input_size=256, crop_size=224, train=True, transform=train_transforms)
    test_data = CUB200(root=config.ROOT, input_size=480, crop_size=448, train=False, transform=test_transforms)
    print('load {} train images!'.format(len(train_data)))
    print('load {} test images!'.format(len(test_data)))

    # wrap to dataloader
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.WORKERS, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.WORKERS, pin_memory=True)

    # define optimizer
    from optimizer import PolyOptimizer
    max_step = len(train_data) // config.BATCH_SIZE * config.EPOCHS
    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': config.LR, 'weight_decay': config.WEIGHT_DECAY},
        {'params': param_groups[1], 'lr': 2 * config.LR, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * config.LR, 'weight_decay': config.WEIGHT_DECAY},
        {'params': param_groups[3], 'lr': 20 * config.LR, 'weight_decay': 0}
    ], lr=config.LR, weight_decay=config.WEIGHT_DECAY, max_step=max_step)

    num_iters = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters * config.EPOCHS)

    start_epoch = 0

    # training part
    for epoch in range(start_epoch, config.EPOCHS):
        # training
        train(config, train_loader, model, criterion, optimizer, epoch, scheduler)

        # testing
        best_CorLoc, best_threshold = test(config, test_loader, model, criterion, epoch)

        torch.save(
            {"state_dict": model.state_dict(),
             "epoch": epoch + 1,
             "CorLoc": best_CorLoc,
             "Threshold": best_threshold,
             "Flag": flag,
             }, '{}/checkpoints/{}/current_epoch.pth'.format(config.DEBUG, config.EXPERIMENT))

    print('Training finished...')
    print('--------------------')


def train(config, train_loader, model, criterion, optimizer, epoch, scheduler):

    # set up the averagemeters
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_bg_bg = AverageMeter()
    losses_bg_fg = AverageMeter()
    losses_fg_fg = AverageMeter()
    global flag
    # switch to train mode
    model.train()
    # record time
    end = time.time()

    # training step
    for i, (input, target, cls_name, img_name) in enumerate(train_loader):

        # data to gpu
        input = input.cuda()

        optimizer.zero_grad()
        fg_feats, bg_feats, ccam = model(input)

        loss1 = criterion[0](bg_feats)
        loss2 = criterion[1](bg_feats, fg_feats)
        loss3 = criterion[2](fg_feats)
        loss = loss1 + loss2 + loss3

        loss.backward()
        optimizer.step()

        losses.update(loss.data.item(), input.size(0))
        losses_bg_bg.update(loss1.data.item(), input.size(0))
        losses_bg_fg.update(loss2.data.item(), input.size(0))
        losses_fg_fg.update(loss3.data.item(), input.size(0))

        if epoch == 0 and i == 300:
            flag = check_positive(ccam)
            print(f"Is Negative: {flag}")
        if flag:
            ccam = 1 - ccam

        # measure elapsed time
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print the current status
        if i % config.PRINT_FREQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 'BG-C-BG {loss_bgbg.val:.4f} ({loss_bgbg.avg:.4f})\t'
                 'BG-C-FG {loss_bgfg.val:.4f} ({loss_bgfg.avg:.4f})\t'
                 'FG-C-FG {loss_fg_fg.val:.4f} ({loss_fg_fg.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, loss_bgbg=losses_bg_bg, loss_bgfg=losses_bg_fg, loss_fg_fg=losses_fg_fg), flush=True)

            # image debug
            visualize_heatmap(config, config.EXPERIMENT, input.clone().detach(), ccam, cls_name, img_name)

    # print the learning rate
    lr = scheduler.get_last_lr()[0]
    print("Epoch {:d} finished with lr={:f}".format(epoch + 1, lr))


def test(config, test_loader, model, criterion, epoch):

    # set up the averagemeters
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_bg_bg = AverageMeter()
    losses_bg_fg = AverageMeter()
    losses_fg_fg = AverageMeter()
    threshold = [(i + 1) / config.NUM_THRESHOLD for i in range(config.NUM_THRESHOLD - 1)]
    print('current threshold list: {}'.format(threshold))

    # switch to evaluate mode
    model.eval()
    global flag
    # record the time
    end = time.time()

    total = 0
    Corcorrect = torch.Tensor([[0] for i in range(len(threshold))])

    # testing
    with torch.no_grad():
        for i, (input, target, bboxes, cls_name, img_name) in enumerate(test_loader):

            # data to gpu
            input = input.cuda()

            # inference the model
            fg_feats, bg_feats, ccam = model(input)

            if flag:
                ccam = 1 - ccam

            pred_boxes_t = [[] for j in range(len(threshold))]  # x0,y0, x1, y1
            for j in range(input.size(0)):

                estimated_boxes_at_each_thr, _ = compute_bboxes_from_scoremaps(
                    ccam[j, 0, :, :].detach().cpu().numpy().astype(np.float32), threshold, input.size(-1)/ccam.size(-1), multi_contour_eval=False)

                for k in range(len(threshold)):
                    pred_boxes_t[k].append(estimated_boxes_at_each_thr[k])

            loss1 = criterion[0](bg_feats)            # bg contrast bg
            loss2 = criterion[1](bg_feats, fg_feats)  # fg contrast fg
            loss3 = criterion[2](fg_feats)            # fg contrast fg
            loss = loss1 + loss2 + loss3

            # acc1 = accuracy(main_out.data, target)[0]
            losses.update(loss.data.item(), input.size(0))
            losses_bg_bg.update(loss1.data.item(), input.size(0))
            losses_bg_fg.update(loss2.data.item(), input.size(0))
            losses_fg_fg.update(loss3.data.item(), input.size(0))

            # measure elapsed time
            torch.cuda.synchronize()

            total += input.size(0)
            for j in range(len(threshold)):
                pred_boxes = pred_boxes_t[j]
                pred_boxes = torch.from_numpy(np.array([pred_boxes[k][0] for k in range(len(pred_boxes))])).float()
                gt_boxes = bboxes[:, 1:].float()

                # calculate
                inter = intersect(pred_boxes, gt_boxes)
                area_a = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
                area_b = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                union = area_a + area_b - inter
                IOU = inter / union
                IOU = torch.where(IOU <= 0.5, IOU, torch.ones(IOU.shape[0]))
                IOU = torch.where(IOU > 0.5, IOU, torch.zeros(IOU.shape[0]))

                Corcorrect[j] += IOU.sum()

            batch_time.update(time.time() - end)
            end = time.time()

            # print the current testing status
            if i % config.PRINT_FREQ == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'BG-C-BG {loss_bgbg.val:.4f} ({loss_bgbg.avg:.4f})\t'
                     'BG-C-FG {loss_bgfg.val:.4f} ({loss_bgfg.avg:.4f})\t'
                     'FG-C-FG {loss_fg_fg.val:.4f} ({loss_fg_fg.avg:.4f})'.format(
                    epoch, i, len(test_loader), batch_time=batch_time,
                    loss=losses, loss_bgbg=losses_bg_bg, loss_bgfg=losses_bg_fg, loss_fg_fg=losses_fg_fg), flush=True)

                # image debug
                visualize_heatmap(config, config.EXPERIMENT, input.clone().detach(), ccam, cls_name, img_name, phase='test', bboxes=pred_boxes_t[config.NUM_THRESHOLD // 2], gt_bboxes=bboxes)

    current_best_CorLoc = 0
    current_best_CorLoc_threshold = 0
    for i in range(len(threshold)):
        if (Corcorrect[i].item() / total) * 100 > current_best_CorLoc:
            current_best_CorLoc = (Corcorrect[i].item() / total) * 100
            current_best_CorLoc_threshold = threshold[i]

    print('Current => Correct: {:.2f}, threshold: {}'.format(current_best_CorLoc, current_best_CorLoc_threshold))

    return current_best_CorLoc, current_best_CorLoc_threshold


if __name__ == '__main__':
    main()
