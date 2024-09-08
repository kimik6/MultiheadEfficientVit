from IOUEval import SegmentationMetric
import logging
import logging.config
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import transforms as T
import pickle

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def resize(
        x: torch.Tensor,
        size: any or None = None,
        scale_factor: list[float] or None = None,
        mode: str = "bicubic",
        align_corners: bool or None = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


LOGGING_NAME = "custom"


def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level, }},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False, }}})


set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        return IoU[1]

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        # print(imgLabel.shape)
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

def poly_lr_scheduler(args, optimizer, epoch, power=2):
    lr = round(args.lr * (1 - epoch / args.max_epochs) ** power, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

loss_total = AverageMeter()
tversky_loss_total = AverageMeter()
focal_loss_total = AverageMeter()
loss_adv_total = AverageMeter()
loss_D_target_total = AverageMeter()
loss_D_source_total = AverageMeter()

all_logs = [] 

def train(args, data_loader, model, criterion,  optimizer, epoch):
    device = args.device

    loss_total.reset()
    tversky_loss_total.reset()
    focal_loss_total.reset()

    total_batches = len(data_loader)
    train_loader = enumerate(data_loader)
    # pbar = enumerate(zip(source_loader, cycle(target_loader)))
    LOGGER.info(('\n' + '%13s' * 4) % ('Epoch', 'TverskyLoss', 'FocalLoss',  'Total Loss' ))
    pbar = (tqdm(train_loader, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}'))
    for i, (source_data) in pbar:
        optimizer.zero_grad()

        (_, train_input, labels) = source_data

        if args.device == 'cuda:0':
            train_input = train_input.cuda().float()
            labels[0] = labels[0].cuda()
            labels[1] = labels[1].cuda()

        train_output = model(train_input)
        train_output_resized = (resize(train_output[0], [512, 512]), resize(train_output[1], [512, 512]))

        focal_loss, tversky_loss, loss = criterion(train_output_resized, labels)
        loss_total.update(loss,args.batch_size)
        tversky_loss_total.update(tversky_loss,args.batch_size)
        focal_loss_total.update(focal_loss,args.batch_size)
        loss.backward()
        optimizer.step()
        logs = {
            "epoch": epoch,
            "tversky_loss": tversky_loss_total.avg,
            "focal_loss": focal_loss_total.avg,
            "loss_total": loss_total.avg,
        }
        
        all_logs.append(logs)  # Append the logs for this epoch to the list
        
        # Save all_logs to the pickle file
        with open('fine_tune_train_logs.pkl', 'wb') as f:
            pickle.dump(all_logs, f)

        pbar.set_description(('%13s' * 1 + '%13.4g' * 3) %
                             (f'{epoch}/{args.max_epochs - 1}', tversky_loss_total.avg, focal_loss_total.avg, loss_total.avg))
@torch.no_grad()
def val(val_loader, model,multi):
    # os.mkdir('/kaggle/working/outputs')

    model.eval()

    DA = SegmentationMetric(2)
    LL = SegmentationMetric(2)

    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()

    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=total_batches)
    for i, (_, input, target) in pbar:
        input = input.cuda().float()
        # target = target.cuda()

        input_var = input
        DA.reset()
        DA.addBatch(da_predict.cpu(), da_gt.cpu())

        da_acc = DA.pixelAccuracy()
        da_IoU = DA.IntersectionOverUnion()
        da_mIoU = DA.meanIntersectionOverUnion()

        with torch.no_grad():
            output = model(input_var)
            # output = model(input_var)
            if multi == 'multi':
                output = (resize(output[0], [512, 512]), resize(output[1], [512, 512]))
                out_da, out_ll = output
                target_da, target_ll = target

                _, da_gt = torch.max(target_da, 1)
                _, da_predict = torch.max(out_da, 1)

                _, ll_predict = torch.max(out_ll, 1)
                _, ll_gt = torch.max(target_ll, 1)

                DA.reset()
                DA.addBatch(da_predict.cpu(), da_gt.cpu())

                da_acc = DA.pixelAccuracy()
                da_IoU = DA.IntersectionOverUnion()
                da_mIoU = DA.meanIntersectionOverUnion()

                da_acc_seg.update(da_acc, input.size(0))
                da_IoU_seg.update(da_IoU, input.size(0))
                da_mIoU_seg.update(da_mIoU, input.size(0))

                LL.reset()
                LL.addBatch(ll_predict.cpu(), ll_gt.cpu())

                ll_acc = LL.pixelAccuracy()
                ll_IoU = LL.IntersectionOverUnion()
                ll_mIoU = LL.meanIntersectionOverUnion()

                ll_acc_seg.update(ll_acc, input.size(0))
                ll_IoU_seg.update(ll_IoU, input.size(0))
                ll_mIoU_seg.update(ll_mIoU, input.size(0))
                
            elif multi == 'lane':
                 output = (resize(output, [512, 512]))
                 out_ll = output
                 _,target_ll = target
                 _, ll_predict = torch.max(out_ll, 1)
                 _, ll_gt = torch.max(target_ll, 1)
                 LL.reset()
                 LL.addBatch(ll_predict.cpu(), ll_gt.cpu())

                 ll_acc = LL.pixelAccuracy()
                 ll_IoU = LL.IntersectionOverUnion()
                 ll_mIoU = LL.meanIntersectionOverUnion()

                 ll_acc_seg.update(ll_acc, input.size(0))
                 ll_IoU_seg.update(ll_IoU, input.size(0))
                 ll_mIoU_seg.update(ll_mIoU, input.size(0))
            
            else:
                 output = (resize(output, [512, 512]))
                 out_da = output 
                 target_da,_ = target
                 _, da_gt = torch.max(target_da, 1)
                 _, da_predict = torch.max(out_da, 1)   
                
                 DA.reset()
                 DA.addBatch(da_predict.cpu(), da_gt.cpu())

                 da_acc = DA.pixelAccuracy()
                 da_IoU = DA.IntersectionOverUnion()
                 da_mIoU = DA.meanIntersectionOverUnion()

                 da_acc_seg.update(da_acc, input.size(0))
                 da_IoU_seg.update(da_IoU, input.size(0))
                 da_mIoU_seg.update(da_mIoU, input.size(0))             

        
    if multi == ' multi':
        da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
        ll_segment_result = (ll_acc_seg.avg, ll_IoU_seg.avg, ll_mIoU_seg.avg)
        return da_segment_result, ll_segment_result
    elif multi == 'lane':
        ll_segment_result = (ll_acc_seg.avg, ll_IoU_seg.avg, ll_mIoU_seg.avg)
        return ll_segment_result
    else:
        da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
        return da_segment_result



def valid(mymodel, Dataset,multi):
    '''
    Main function for trainign and validation
    :param args: global arguments
    :return: None
    '''

    # load the model
    model = mymodel.eval()
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True

    valLoader = torch.utils.data.DataLoader(
        Dataset,
        batch_size=32, shuffle=False, num_workers=1, pin_memory=True)

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    #     model.load_state_dict(torch.load(PATH))
    model.eval()
    example = torch.rand(16, 3, 512, 512).cuda()
    model = torch.jit.trace(model, example)
    if multi == 'multi':
        da_segment_results, ll_segment_results = val(valLoader, model)
        msg = '\n Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n'.format(
        da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],)
        print(msg)

        msg2 = '\n lane line detection: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})    mIOU({ll_seg_miou:.3f})\n'.format(
            ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2])
        print(msg2)

    elif multi == 'lane':
        ll_segment_results = val(valLoader, model)
        msg2 = '\n lane line detection: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})    mIOU({ll_seg_miou:.3f})\n'.format(
            ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2])
        print(msg2)
    else:
        da_segment_results = val(valLoader, model)
        msg = '\n Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n'.format(
        da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],)
        print(msg)


    return da_segment_results[2],ll_segment_results[1]


def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)


def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])


def show_grays(images, cols=2):
    plt.rcParams['figure.figsize'] = (20, 15)
    imgs = images['image'] if isinstance(images, dict) else images

    if not isinstance(imgs, type([])):
        imgs = [imgs]
    fix, ax = plt.subplots(ncols=cols, nrows=np.ceil(len(imgs) / cols).astype(np.int8), squeeze=False)
    for i, img in enumerate(imgs):
        ax[i // cols, i % cols].imshow(np.asarray(img), cmap='gray')
        ax[i // cols, i % cols].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if isinstance(images, dict): ax[i // cols, i % cols].title.set_text(images['name'][i])
    plt.show()


def show_seg_result(img, result, index, epoch, save_dir=None, is_ll=False, palette=None, is_demo=False, is_gt=False,
                    outsize=(1280, 720)):
    # img = mmcv.imread(img)
    # img = img.copy()
    # seg = result[0]
    if palette is None:
        palette = np.random.randint(
            0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3  # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2

    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)

        # for label, color in enumerate(palette):
        #     color_area[result[0] == label, :] = color

        color_area[result[0] == 1] = [51, 153, 102]
        color_area[result[1] == 1] = [153, 51, 102]
        color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, outsize, interpolation=cv2.INTER_LINEAR)

    if not is_demo:
        if not is_gt:
            if not is_ll:
                cv2.imwrite(save_dir + "/batch_{}_{}_da_segresult.png".format(epoch, index), img)
            else:
                cv2.imwrite(save_dir + "/batch_{}_{}_ll_segresult.png".format(epoch, index), img)
        else:
            if not is_ll:
                cv2.imwrite(save_dir + "/batch_{}_{}_da_seg_gt.png".format(epoch, index), img)
            else:
                cv2.imwrite(save_dir + "/batch_{}_{}_ll_seg_gt.png".format(epoch, index), img)
    return img

invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                T.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

def plot_the_last_epoch(batch,segmentations):
    vis_idx = 0
    for vis_idx in range(32):
        x = batch[1]
        seg = batch[2][0]
        ll = batch[2][1]
        y_seg_pred ,y_ll_pred = segmentations
        y_seg_pred = resize(y_seg_pred, [512, 512])
        y_ll_pred = resize(y_ll_pred, [512, 512])

        vis_pred1 = (y_seg_pred)[vis_idx][1].detach().cpu().numpy()
        vis_pred2 = (y_seg_pred)[vis_idx][0].detach().cpu().numpy()
        vis_pred3 = (y_ll_pred)[vis_idx][1].detach().cpu().numpy()
        vis_pred4 = (y_ll_pred)[vis_idx][0].detach().cpu().numpy()

        vis_logit = (y_seg_pred)[vis_idx].argmax(0).detach().cpu().numpy()
        vis_logit2 = (y_ll_pred)[vis_idx].argmax(0).detach().cpu().numpy()

        vis_input = invTrans(x[vis_idx]).permute(1, 2, 0).cpu().numpy()
        vis_input = cv2.cvtColor(vis_input, cv2.COLOR_BGR2RGB)

        vis_label1 = seg[vis_idx][1].long().detach().cpu().numpy()
        vis_label2 = ll[vis_idx][1].long().detach().cpu().numpy()

        viss = [vis_pred1, vis_pred2, vis_pred3, vis_pred4, vis_logit, vis_logit2, vis_label1, vis_label2, vis_input]
        show_grays(viss, 3)

        img_det1 = show_seg_result(vis_input * 255, (vis_logit, vis_logit2), 0, 0, is_demo=True)
        img_det2 = show_seg_result(vis_input * 255, (vis_label1, vis_label2), 0, 0, is_demo=True)

        show_grays([img_det1, img_det2])