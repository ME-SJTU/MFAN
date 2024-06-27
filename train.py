import argparse

from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from metrics import *
from model import *
from utils import *
import pickle
import torch
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--dataset', default='eth', help='eth,hotel,univ,zara1,zara2')

parser.add_argument('--rgcn_num', type=int, default=3, help='mini-batch size')

parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--clip_grad', type=float, default=10, help='gadient clipping')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum of lr')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay on l2 reg')
parser.add_argument('--lr_sh_rate', type=int, default=100, help='number of steps to drop the lr')
parser.add_argument('--milestones', type=int, default=[50, 100], help='number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=True, help='Use lr rate scheduler')
parser.add_argument('--tag', default='domain_mfan', help='personal tag for the model ')
parser.add_argument('--gpu_num', default="0", type=str)

args = parser.parse_args()

print("Training initiating....")
print(args)


def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred, V_target)


metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999, 'min_train_epoch': -1,
                    'min_train_loss': 9999999999999999}


def data_loader():
    """Data prepared"""
    obs_seq_len = args.obs_len
    pred_seq_len = args.pred_len
    data_set = './dataset/' + args.dataset + '/'

    """Training dataset loading"""
    data_train = TrajectoryDataset(
        data_set + 'train/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_train = DataLoader(
        data_train,
        batch_size=1,
        shuffle=True,
        num_workers=0)

    """Validating dataset loading"""
    data_val = TrajectoryDataset(
        # data_set + 'val/',
        './dataset/hotel/val01/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_val = DataLoader(
        data_val,
        batch_size=1,
        shuffle=True,
        num_workers=1)

    return loader_train, loader_val


def train(epoch, model, optimizer, checkpoint_dir, loader_train):
    global metrics, constant_metrics
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_train):
        batch_count += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]

        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel, v_obs, v_gt, obs_spat_edge, obs_temp_edge, \
        spat_mask, temp_mask = batch

        # obs_traj -- observed absolute coordinate [1 obs_len N 2]
        # pred_traj_gt -- ground truth absolute coordinate [1 pred_len N 2]
        # v_obs -- velocity of observed trajectory [1 obs_len N 2]
        # v_gt -- velocity of ground-truth [1 pred_len N 2]
        # obs_spat_edge -- spatial edge features [1 obs_len N N 3]
        # obs_temp_edge -- temporal edge features [1 N obs_len obs_len 3]
        # spat_mask -- spatial edge interaction mask [1 obs_len N N 3] 1 -- be edge 0 -- not edge
        # temp_mask -- temporal edge interaction mask [1 N obs_len obs_len 3]

        spat_self_connection = torch.ones((v_obs.shape[1], v_obs.shape[2], v_obs.shape[2]), device='cuda') * \
                           torch.eye(v_obs.shape[2], device='cuda')  # [obs_len N N]
        temp_self_connection = torch.ones((v_obs.shape[2], v_obs.shape[1], v_obs.shape[1]), device='cuda') * \
                            torch.eye(v_obs.shape[1], device='cuda')  # [N obs_len obs_len]
        self_connection = [spat_self_connection, temp_self_connection]

        st_mask = [spat_mask, temp_mask]

        optimizer.zero_grad()

        v_pred = model(v_obs, obs_spat_edge, obs_temp_edge, st_mask)  # A_obs <8, #, #>
        v_pred = v_pred.squeeze()

        v_gt = v_gt.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(v_pred, v_gt)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            # Metrics
            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)
    metrics['train_loss'].append(loss_batch / batch_count)

    if metrics['train_loss'][-1] < constant_metrics['min_train_loss']:
        constant_metrics['min_train_loss'] = metrics['train_loss'][-1]
        constant_metrics['min_train_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'train_best.pth')  # OK


def vald(epoch, model, checkpoint_dir, loader_val):
    global metrics, constant_metrics
    model.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_val):
        batch_count += 1
        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel, v_obs, v_gt, \
        obs_spat_edge, obs_temp_edge, spat_mask, temp_mask = batch

        with torch.no_grad():
            spat_self_connection = torch.ones((v_obs.shape[1], v_obs.shape[2], v_obs.shape[2]), device='cuda') * \
                                   torch.eye(v_obs.shape[2], device='cuda')  # [obs_len N N]
            temp_self_connection = torch.ones((v_obs.shape[2], v_obs.shape[1], v_obs.shape[1]), device='cuda') * \
                                   torch.eye(v_obs.shape[1], device='cuda')  # [N obs_len obs_len]
            self_connection = [spat_self_connection, temp_self_connection]

            st_mask = [spat_mask, temp_mask]

            # v_obs = v_obs[:, :, :, 1:]
            # v_gt = v_gt[:, :, :, 1:]

            v_pred = model(v_obs, obs_spat_edge, obs_temp_edge, st_mask)  # A_obs <8, #, #>
            v_pred = v_pred.squeeze()
            v_gt = v_gt.squeeze()

            if batch_count % args.batch_size != 0 and cnt != turn_point:
                l = graph_loss(v_pred, v_gt)

                if is_fst_loss:
                    loss = l
                    is_fst_loss = False
                else:
                    loss += l

            else:
                loss = loss / args.batch_size
                is_fst_loss = True
                # Metrics
                loss_batch += loss.item()
                print('VALD:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)
    metrics['val_loss'].append(loss_batch / batch_count)

    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')  # OK


def main(args):

    loader_train, loader_val = data_loader()

    print('Training started ...')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    model = PredictionModel(embedding_dims=32, dropout=0, obs_len=8,
                            pred_len=12, num_tcn=5, out_dims=5, conv_num=5, gat_num=args.rgcn_num).cuda()

    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    # if args.use_lrschd:
    #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.5)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

    if args.use_lrschd:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.5)

    # if args.use_lrschd:
    #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer)

    checkpoint_dir = './checkpoints/' + args.tag + '/' + args.dataset + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    for epoch in range(args.num_epochs):
        train(epoch, model, optimizer, checkpoint_dir, loader_train)
        vald(epoch, model, checkpoint_dir, loader_val)

        if args.use_lrschd:
            scheduler.step()

        print('*' * 30)
        print('Epoch:', args.dataset + '/' + args.tag, ":", epoch)
        for k, v in metrics.items():
            if len(v) > 0:
                print(k, v[-1])

        print(constant_metrics)
        print('*' * 30)

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)

        print("*" * 50)

    plt.plot(np.arange(len(metrics['train_loss'])), metrics['train_loss'],label="train loss")
    plt.plot(np.arange(len(metrics['val_loss'])), metrics['val_loss'], label="valid loss")
    plt.legend() #显示图例
    plt.xlabel('epoches')
    plt.title('Model loss')
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
