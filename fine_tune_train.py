import os
import sys

# put the directory efficientvit instead of '..'

######
import torch

from argparse import ArgumentParser
from utils import train, valid, save_checkpoint, poly_lr_scheduler
import torch.optim.lr_scheduler
from torchvision.transforms import transforms as T
import DataSet as myDataLoader
from loss import TotalLoss
import os
import torch.backends.cudnn as cudnn
from model.seg_model_zoo import create_seg_model
import pickle
all_logs=[]
def train_net(args):
    # load the model
    cuda_available = torch.cuda.is_available()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

    ])
    da_seg_miou=0
    ll_seg_iou=0
    pretrained = args.pretrained
    backbone_weight_url = args.backbone_weight_url ,
    engine = args.engine
    if pretrained is not None:
        model = create_seg_model(args.model, 'bdd',args.task,weight_url=pretrained)
    elif backbone_weight_url is not None:
        model = create_seg_model(args.model, 'bdd',args.task,backbone_weight_url=backbone_weight_url,weight_url=None)
    else:
        model = create_seg_model(args.model, 'bdd', False)

    args.savedir = args.savedir + '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    if cuda_available:
        args.onGPU = True
        model = model.cuda()

        cudnn.benchmark = True

    criteria = TotalLoss(device=args.device)
    start_epoch = 0
    lr = args.lr

    # optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

    # optimizer.zero_grad()
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, 0))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

    optimizer.zero_grad()

    target_valLoader = myDataLoader.MyDataset(transform=transform, valid=True, engin=engine, data='IADD')

    source_valLoader = myDataLoader.MyDataset(transform=transform, valid=True, engin=engine, data='bdd')

    source_loader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(transform=transform, valid=False, engin=engine, data='bdd'),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    target_loader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(transform=transform, valid=False, engin=engine, data=args.data),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    model.train()
    if args.model == 'b0':
        for param in model.backbone.input_stem.parameters():
            param.requires_grad = False
    if args.task == 'lane':
        for param in model.head1.parameters():
            param.requires_grad = False
    elif args.task == 'drivable':
        for param in model.head2.parameters():
            param.requires_grad = False
    # ct = 0
    # for child in model.backbone.stages.children():
    #     ct += 1
    #     if ct == 5:
    #         for param in child.parameters():
    #             param.requires_grad = False
    for epoch in range(start_epoch, args.max_epochs):

        model_file_name = args.savedir + os.sep + 'model_{}.pth'.format(epoch)

        checkpoint_file_name = args.savedir + os.sep + 'checkpoint_{}.pth.tar'.format(epoch)
        poly_lr_scheduler(args, optimizer, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " + str(lr))
        # train for one epoch
        
        train(args,target_loader, model, criteria, optimizer, epoch)

        if args.data == 'bdd':
            if args.task == 'multi':
                da_seg_miou,ll_seg_iou = valid(model, source_valLoader,args.task,args.model)
            elif args.task == 'lane':
                ll_seg_iou = valid(model, source_valLoader,args.task,args.model)
            elif args.task == 'drivable':
                da_seg_miou = valid(model, source_valLoader,args.task,args.model)

        elif args.data == 'IADD':
            if args.task == 'multi':
                da_seg_miou,ll_seg_iou = valid(model, target_valLoader,args.task,args.model)
            elif args.task == 'lane':
                ll_seg_iou = valid(model, target_valLoader,args.task,args.model)
            elif args.task == 'drivable':
                da_seg_miou = valid(model, target_valLoader,args.task,args.model)


                
            

        logs = {
            "epoch": epoch,
            "drivable area miou": da_seg_miou,
            "lane line iou": ll_seg_iou
            }
        
        all_logs.append(logs)  # Append the logs for this epoch to the list
        
        # Save all_logs to the pickle file
        with open('fine_tune_eval_logs.pkl', 'wb') as f:
            pickle.dump(all_logs, f)

        torch.save(model.state_dict(), model_file_name)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr
        }, checkpoint_file_name)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model', type=str, default='b0', help='Choose between b0, b1, b2 or b3 efficientVit backbones')
    parser.add_argument('--backbone_weight_url', type=str, default=None, help='input the backbone weights path for your selected model')
    parser.add_argument('--max_epochs', type=int, default=10, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=2e-7, help='Initial learning rate')
    parser.add_argument('--savedir', default='./test_', help='directory to save the results')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')
    parser.add_argument('--pretrained', default='./pretrained/pretrained_bdd.pth', help='Pretrained ESPNetv2 weights.')
    parser.add_argument('--engine', default='kaggle', help='choose youre prefered engine, kaggle or colab.')
    parser.add_argument('--data', default='bdd', help='DA mode or DAST mode?.')
    parser.add_argument('--task', default='multi', help='"multi" or "lane" or "drivable" segmentation task.')

    train_net(parser.parse_args())
