import torch
import torch
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import val, netParams
import torch.optim.lr_scheduler
from const import *
from torchvision.transforms import transforms as T
from model.seg_model_zoo import create_seg_model

def validation(args):
    '''
    Main function for trainign and validation
    :param args: global arguments
    :return: None
    '''
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

    ])
    # load the model
    pretrained = args.pretrained
    model = create_seg_model('b0', 'bdd', weight_url=pretrained)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(transform=transform, valid=True, engin=args.engine, data=args.data),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    model.eval()
    example = torch.rand(12, 3, 512, 512).cuda()
    model = torch.jit.trace(model, example)
    da_segment_results, ll_segment_results = val(valLoader, model)

    msg = 'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
          'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})'.format(
        da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],
        ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2])
    print(msg)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pretrained', default="./pretrained/pretrained_bdd.pth")
    parser.add_argument('--num_workers', type=int, default=12, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--data', type=str, default='bdd', help='data type, bdd or IADD')
    parser.add_argument('--engine', type=str, default='kaggle', help='No. of parallel threads')

    validation(parser.parse_args())
