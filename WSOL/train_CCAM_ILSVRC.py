"""
ref. from https://github.com/zxhuang1698/interpretability-by-parts
ref. from https://github.com/Sierkinhane/ORNet
modified by sierkinhane
"""
import argparse
import time
import torch.backends.cudnn as cudnn
from utils import *
from dataset.ilsvrc import *
from models.loss import *
import random
from models.model import *
from torchvision import transforms
import yaml
from easydict import EasyDict as edict
import torch.distributed as dist
import torch.multiprocessing as mp

# benchmark before running
cudnn.benchmark = True


def parse_arg():
    parser = argparse.ArgumentParser(description="train CCAM on ILSRVC dataset")
    parser.add_argument('--cfg', type=str, default='config/CCAM_ILSVRC.yaml',
                        help='experiment configuration filename')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--port', type=int, default=2345)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--experiment', type=str, required=True, help='record different experiments')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='adopt different pretrained parameters, [supervised, mocov2, detco]')
    parser.add_argument('--evaluate', type=bool, default=False, help='evaluation mode')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f)
        config = edict(config)
    config.EXPERIMENT = args.experiment
    config.EVALUTATE = args.evaluate
    config.PORT = args.port
    config.LR = args.lr
    config.ALPHA = args.alpha
    config.EPOCHS = args.epoch
    config.SEED = args.seed
    config.PRETRAINED = args.pretrained

    return config, args


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    config, args = parse_arg()
    config.BATCH_SIZE = args.batch_size
    if config.SEED != -1:
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)

    # create folder for logging
    creat_folder(config, args)

    # log
    sys.stdout = Logger('{}/{}_log.txt'.format(config.LOG_DIR, config.EXPERIMENT))

    config.nprocs = torch.cuda.device_count()
    print(config.nprocs, 'processes!')
    mp.spawn(main_worker, nprocs=config.nprocs, args=(config.nprocs, config, args))


flag = True
def main_worker(local_rank, nprocs, config, args):
    global flag
    dist.init_process_group(backend='nccl', init_method=f'tcp://127.0.0.1:{config.PORT}', world_size=nprocs,
                            rank=local_rank)

    # create model
    print("=> creating model...")
    print("alphas is :", config.ALPHA)
    model = get_model(config.PRETRAINED)
    param_groups = model.get_parameter_groups()



    # if local_rank == 0:
        # model_info(model)

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)

    config.BATCH_SIZE = int(config.BATCH_SIZE / nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # define loss function (criterion) and optimizer
    criterion = [SimMaxLoss(metric='cos', alpha=config.ALPHA).cuda(local_rank), SimMinLoss(metric='cos').cuda(local_rank),
                 SimMaxLoss(metric='cos', alpha=config.ALPHA).cuda(local_rank)]

    # data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(size=(480, 480)),
        transforms.CenterCrop(size=(448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # wrap to dataset
    train_data = ILSVRC2012(root=config.ROOT, input_size=256, crop_size=224, train=True, transform=train_transforms)
    test_data = ILSVRC2012(root=config.ROOT, input_size=480, crop_size=448, train=False, transform=test_transforms)
    if local_rank == 0:
        print('load {} train images!'.format(len(train_data)))
        print('load {} test images!'.format(len(test_data)))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)

    # wrap to dataloader
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.BATCH_SIZE,
        num_workers=config.WORKERS, pin_memory=False, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.BATCH_SIZE,
        num_workers=config.WORKERS, pin_memory=True, collate_fn=my_collate, sampler=test_sampler)

    # define optimizer
    from optimizer import PolyOptimizer
    max_step = len(train_data) // config.BATCH_SIZE * config.EPOCHS
    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': config.LR, 'weight_decay': config.WEIGHT_DECAY},
        {'params': param_groups[1], 'lr': 2 * config.LR, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * config.LR, 'weight_decay': config.WEIGHT_DECAY},
        {'params': param_groups[3], 'lr': 20 * config.LR, 'weight_decay': 0}
    ], lr=config.LR, weight_decay=config.WEIGHT_DECAY, max_step=max_step)

    if config.EVALUTATE:
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        model.module.load_state_dict(checkpoint['state_dict'])
        threshold = checkpoint['Threshold']
        flag = checkpoint['Flag']
        if local_rank == 0:
            print("=> loaded best threshold: {}".format(threshold))

        test_sampler.set_epoch(0)
        # evaluate on test set
        evaluate(config, test_loader, model, criterion, threshold, flag, local_rank, nprocs)

        return

    num_iters = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters * config.EPOCHS)

    # optionally resume from a checkpoint
    start_epoch = 0
    # training part
    threshold = 0
    for epoch in range(start_epoch, config.EPOCHS):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)

        # training
        train(config, train_loader, model, criterion, optimizer, epoch, scheduler,
              local_rank, nprocs)
        best_CorLoc, best_threshold = test(config, test_loader, model, criterion, epoch, local_rank, nprocs)
        threshold = best_threshold

        if local_rank == 0:
            torch.save(
                {"state_dict": model.module.state_dict(),
                 "epoch": epoch + 1,
                 "CorLoc": best_CorLoc,
                 "Threshold": best_threshold,
                 "Flag": flag,
                 }, '{}/checkpoints/{}/current_epoch_{}.pth'.format(config.DEBUG, config.EXPERIMENT, epoch + 1))

    print('Training finished...')
    # evaluate on test set

    if local_rank == 0:
        print('Use the last checkpoint for evaluation...')
    test_sampler.set_epoch(0)
    evaluate(config, test_loader, model, criterion, threshold, local_rank, nprocs)

    if local_rank == 0:
        print('Extracting class-agnostic bboxes using best threshold...')
        print('--------------------------------------------------------')

    # extract class-agnostic bboxes
    train_transforms = transforms.Compose([
        transforms.Resize(size=(480, 480)),
        transforms.CenterCrop(size=(448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_data = ILSVRC2012(root=config.ROOT, input_size=480, crop_size=448, train=True, transform=train_transforms)
    if local_rank == 0:
        print('load {} train images!'.format(len(train_data)))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

    # wrap to dataloader
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.BATCH_SIZE,
        num_workers=config.WORKERS, pin_memory=False, sampler=train_sampler)

    extract(config, train_loader, model, threshold, flag, local_rank)

    if local_rank == 0:
        torch.save(
            {"state_dict": model.module.state_dict(),
             "CorLoc": best_CorLoc,
             "Threshold": threshold,
             "Flag": flag,
             }, '{}/checkpoints/{}/last_epoch.pth'.format(config.DEBUG, config.EXPERIMENT))


def train(config, train_loader, model, criterion, optimizer, epoch, scheduler, local_rank, nprocs):

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
        input = input.cuda(local_rank, non_blocking=True)

        optimizer.zero_grad()
        fg_feats, bg_feats, ccam = model(input)

        loss1 = criterion[0](bg_feats)            # bg contrast bg
        loss2 = criterion[1](bg_feats, fg_feats)  # fg contrast fg
        loss3 = criterion[2](fg_feats)            # fg contrast fg

        loss = loss1 + loss2 + loss3  # + loss3

        loss.backward()
        optimizer.step()
        scheduler.step()

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, nprocs)
        reduced_loss_bg_bg = reduce_mean(loss1, nprocs)
        reduced_loss_bg_fg = reduce_mean(loss2, nprocs)
        reduced_loss_fg_fg = reduce_mean(loss3, nprocs)

        losses.update(reduced_loss.data.item(), input.size(0))
        losses_bg_bg.update(reduced_loss_bg_bg.data.item(), input.size(0))
        losses_bg_fg.update(reduced_loss_bg_fg.data.item(), input.size(0))
        losses_fg_fg.update(reduced_loss_fg_fg.data.item(), input.size(0))

        # measure elapsed time
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        if epoch == 0 and i == (len(train_loader)-1):
            flag = check_positive(ccam)
            print(f"Is Negative: {flag}")
        if flag:
            ccam = 1 - ccam
        # print the current status
        if i % config.PRINT_FREQ == 0 and local_rank == 0:
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

    if local_rank == 0:
        # print the learning rate
        lr = scheduler.get_last_lr()[0]
        print("Epoch {:d} finished with lr={:f}".format(epoch + 1, lr))


def test(config, test_loader, model, criterion, epoch, local_rank, nprocs):

    # set up the averagemeters
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_bg_bg = AverageMeter()
    losses_bg_fg = AverageMeter()
    losses_fg_fg = AverageMeter()
    threshold = [(i + 1) / config.NUM_THRESHOLD for i in range(config.NUM_THRESHOLD - 1)]
    if local_rank == 0:
        print('current threshold list: {}'.format(threshold))

    # switch to evaluate mode
    model.eval()

    # record the time
    end = time.time()

    total = 0
    Corcorrect = torch.Tensor([[0] for i in range(len(threshold))]).cuda(local_rank)

    # testing
    with torch.no_grad():
        for i, (input, target, bboxes, cls_name, img_name) in enumerate(test_loader):

            # data to gpu
            input = input.cuda(local_rank, non_blocking=True)

            # inference the model
            fg_feats, bg_feats, ccam = model(input)
            if flag:
                ccam = 1 - ccam

            loss1 = criterion[0](bg_feats)            # bg contrast bg
            loss2 = criterion[1](bg_feats, fg_feats)  # fg contrast fg
            loss3 = criterion[2](fg_feats)            # fg contrast fg

            loss = loss1 + loss2 + loss3

            pred_boxes_t = [[] for j in range(len(threshold))]  # x0,y0, x1, y1
            for j in range(input.size(0)):

                estimated_boxes_at_each_thr, _ = compute_bboxes_from_scoremaps(
                    ccam[j, 0, :, :].detach().cpu().numpy().astype(np.float32), threshold, input.size(-1) / ccam.size(-1),
                    multi_contour_eval=False)
                for k in range(len(threshold)):
                    pred_boxes_t[k].append(estimated_boxes_at_each_thr[k])

            total += input.size(0)
            for j in range(len(threshold)):
                pred_boxes = pred_boxes_t[j]
                gt_boxes = [bboxes[k][:, 1:] for k in range(len(bboxes))]

                # calculate
                IOU = IOUFunciton_ILSRVC(pred_boxes, gt_boxes)

                IOU = torch.where(IOU <= 0.5, IOU, torch.ones(IOU.shape[0]))
                IOU = torch.where(IOU > 0.5, IOU, torch.zeros(IOU.shape[0]))

                Corcorrect[j] += IOU.sum()

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, nprocs)
            reduced_loss_bg_bg = reduce_mean(loss1, nprocs)
            reduced_loss_bg_fg = reduce_mean(loss2, nprocs)
            reduced_loss_fg_fg = reduce_mean(loss3, nprocs)

            losses.update(reduced_loss.data.item(), input.size(0))
            losses_bg_bg.update(reduced_loss_bg_bg.data.item(), input.size(0))
            losses_bg_fg.update(reduced_loss_bg_fg.data.item(), input.size(0))
            losses_fg_fg.update(reduced_loss_fg_fg.data.item(), input.size(0))

            for j in range(len(threshold)):
                Corcorrect[j] = reduce_mean(Corcorrect[j], nprocs)

            # measure elapsed time
            torch.cuda.synchronize()

            batch_time.update(time.time() - end)
            end = time.time()

            # print the current testing status
            if i % config.PRINT_FREQ == 0 and local_rank == 0:
                print('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'BG-C-BG {loss_bgbg.val:.4f} ({loss_bgbg.avg:.4f})\t'
                      'BG-C-FG {loss_bgfg.val:.4f} ({loss_bgfg.avg:.4f})\t'
                      'FG-C-FG {loss_fg_fg.val:.4f} ({loss_fg_fg.avg:.4f})'.format(
                    epoch, i, len(test_loader), batch_time=batch_time,
                    loss=losses, loss_bgbg=losses_bg_bg, loss_bgfg=losses_bg_fg, loss_fg_fg=losses_fg_fg), flush=True)
                # image debug
                visualize_heatmap(config, config.EXPERIMENT, input.clone().detach(), ccam, cls_name, img_name,
                                  phase='test', bboxes=pred_boxes_t[config.NUM_THRESHOLD // 2], gt_bboxes=bboxes)

    current_best_CorLoc = 0

    for i in range(len(threshold)):
        if (Corcorrect[i].item() / total) * 100 > current_best_CorLoc:
            current_best_CorLoc = (Corcorrect[i].item() / total) * 100
            current_best_CorLoc_threshold = threshold[i]

    if local_rank == 0:
        print('Current => Correct: {:.2f}, threshold: {}'.format(current_best_CorLoc, current_best_CorLoc_threshold))

    return current_best_CorLoc, current_best_CorLoc_threshold


def evaluate(config, test_loader, model, criterion, threshold, flag, local_rank, nprocs):
    # set up the averagemeters
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_bg_bg = AverageMeter()
    losses_bg_fg = AverageMeter()
    losses_fg_fg = AverageMeter()

    print(f'use threshold: {threshold}')

    # switch to evaluate mode
    model.eval()

    # record the time
    end = time.time()

    total = 0
    Corcorrect = torch.Tensor([[0]]).cuda(local_rank)

    # testing
    with torch.no_grad():
        for i, (input, target, bboxes, cls_name, img_name) in enumerate(test_loader):

            # data to gpu
            input = input.cuda(local_rank, non_blocking=True)

            # inference the model
            fg_feats, bg_feats, ccam = model(input)
            if flag:
                ccam = 1 - ccam
            loss1 = criterion[0](bg_feats)            # bg contrast bg
            loss2 = criterion[1](bg_feats, fg_feats)  # fg contrast fg
            loss3 = criterion[2](fg_feats)            # fg contrast fg

            loss = loss1 + loss2 + loss3  # + loss3

            pred_boxes_t = []  # x0,y0, x1, y1
            for j in range(input.size(0)):
                estimated_boxes_at_each_thr, _ = compute_bboxes_from_scoremaps(
                    ccam[j, 0, :, :].detach().cpu().numpy().astype(np.float32), [threshold], input.size(-1) / ccam.size(-1),
                    multi_contour_eval=False)
                pred_boxes_t.append(estimated_boxes_at_each_thr[0])

            total += input.size(0)

            pred_boxes = pred_boxes_t
            gt_boxes = [bboxes[k][:, 1:] for k in range(len(bboxes))]

            # calculate
            IOU = IOUFunciton_ILSRVC(pred_boxes, gt_boxes)

            IOU = torch.where(IOU <= 0.5, IOU, torch.ones(IOU.shape[0]))
            IOU = torch.where(IOU > 0.5, IOU, torch.zeros(IOU.shape[0]))

            Corcorrect += IOU.sum()
            Corcorrect = reduce_mean(Corcorrect, nprocs)

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, nprocs)
            reduced_loss_bg_bg = reduce_mean(loss1, nprocs)
            reduced_loss_bg_fg = reduce_mean(loss2, nprocs)
            reduced_loss_fg_fg = reduce_mean(loss3, nprocs)

            losses.update(reduced_loss.data.item(), input.size(0))
            losses_bg_bg.update(reduced_loss_bg_bg.data.item(), input.size(0))
            losses_bg_fg.update(reduced_loss_bg_fg.data.item(), input.size(0))
            losses_fg_fg.update(reduced_loss_fg_fg.data.item(), input.size(0))

            # measure elapsed time
            torch.cuda.synchronize()

            batch_time.update(time.time() - end)
            end = time.time()

            # save predicted bboxes
            save_bbox_as_json(config, config.EXPERIMENT, i, local_rank, pred_boxes, cls_name, img_name)

            # print the current testing status
            if i % config.PRINT_FREQ == 0 and local_rank == 0:
                print('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'BG-C-BG {loss_bgbg.val:.4f} ({loss_bgbg.avg:.4f})\t'
                      'BG-C-FG {loss_bgfg.val:.4f} ({loss_bgfg.avg:.4f})\t'
                      'FG-C-FG {loss_fg_fg.val:.4f} ({loss_fg_fg.avg:.4f})'.format(
                    0, i, len(test_loader), batch_time=batch_time,
                    loss=losses, loss_bgbg=losses_bg_bg, loss_bgfg=losses_bg_fg, loss_fg_fg=losses_fg_fg), flush=True)
                # image debug
                visualize_heatmap(config, config.EXPERIMENT, input.clone().detach(), ccam, cls_name, img_name,
                                  phase='test', bboxes=pred_boxes, gt_bboxes=bboxes)

    correct_loc = (Corcorrect.item() / total) * 100

    if local_rank == 0:
        print(
            ' \033[92m* Best  => Correct Loc: {:.2f}, threshold: {} \033[0m'.format(
                correct_loc,
                threshold))

def extract(config, train_loader, model, threshold, flag, local_rank):
    # set up the averagemeters
    batch_time = AverageMeter()

    print(f'use threshold: {threshold}')

    # switch to evaluate mode
    model.eval()

    # record the time
    end = time.time()

    total = 0

    # testing
    with torch.no_grad():
        for i, (input, target, cls_name, img_name) in enumerate(train_loader):

            # data to gpu
            input = input.cuda(local_rank, non_blocking=True)

            # inference the model
            fg_feats, bg_feats, ccam = model(input)
            if flag:
                ccam = 1 - ccam

            pred_boxes_t = []  # x0,y0, x1, y1
            for j in range(input.size(0)):
                estimated_boxes_at_each_thr, _ = compute_bboxes_from_scoremaps(
                    ccam[j, 0, :, :].detach().cpu().numpy().astype(np.float32), [threshold], input.size(-1) / ccam.size(-1),
                    multi_contour_eval=False)
                pred_boxes_t.append(estimated_boxes_at_each_thr[0])

            total += input.size(0)

            pred_boxes = pred_boxes_t

            torch.distributed.barrier()

            # measure elapsed time
            torch.cuda.synchronize()

            batch_time.update(time.time() - end)
            end = time.time()

            # save predicted bboxes
            save_bbox_as_json(config, config.EXPERIMENT, i, local_rank, pred_boxes, cls_name, img_name)

            # print the current testing status
            if i % config.PRINT_FREQ == 0 and local_rank == 0:
                print('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    0, i, len(train_loader), batch_time=batch_time), flush=True)
                # image debug
                visualize_heatmap(config, config.EXPERIMENT, input.clone().detach(), ccam, cls_name, img_name,
                                  phase='train', bboxes=pred_boxes)

if __name__ == '__main__':
    main()
