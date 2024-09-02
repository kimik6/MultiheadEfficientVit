import os

######
import torch
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import train, save_checkpoint, poly_lr_scheduler
import torch.optim.lr_scheduler
from torchvision.transforms import transforms as T
from model.seg_model_zoo import create_seg_model
from loss import TotalLoss



def train_net(args):
    # load the model
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()

    if args.pretrained is not None:
        model = create_seg_model('b0', 'bdd','multi', weight_url=args.pretrained)
    else:
        model = create_seg_model('b0', 'bdd','multi', False)

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    args.savedir = args.savedir + '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

    ])


    val_loader = myDataLoader.MyDataset(transform=transform, valid=True, engin=args.engine, data='bdd')

    train_loader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(transform=transform, valid=False, engin=args.engine, data='bdd'),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)


    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True

    criteria = TotalLoss()

    start_epoch = 0
    lr = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(start_epoch, args.max_epochs):

        model_file_name = args.savedir + os.sep + 'model_{}.pth'.format(epoch)
        checkpoint_file_name = args.savedir + os.sep + 'checkpoint_{}.pth.tar'.format(epoch)
        poly_lr_scheduler(args, optimizer, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " + str(lr))

        # train for one epoch
        model.train()
        model.backbone.required_grad = False
        train(args, train_loader, model, criteria, optimizer, epoch)

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
    parser.add_argument('--max_epochs', type=int, default=100, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--savedir', default='./test_', help='directory to save the results')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')
    parser.add_argument('--pretrained', default='./pretrained/pretrained_bdd.pth', help='Pretrained ESPNetv2 weights.')
    parser.add_argument('--engine', default='kaggle', help='choose youre prefered engine, kaggle or colab.')

    train_net(parser.parse_args())
