import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--algorithm',type=str,default='Fedavg')
# # #System settings
parser.add_argument('--gpu',type=bool,default=True)
parser.add_argument('--device',default='cpu')
parser.add_argument('--user_num',type=int,default=3)
parser.add_argument('--global_epoch',type=int,default=100)
parser.add_argument('--local_epoch',type=int,default=1)
parser.add_argument('--fraction',type=float,default=1)
parser.add_argument('--seed',type=float,default=1)
parser.add_argument('--lr',type=float,default=0.0001)
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--LOSS',type=str,default='MSE')
parser.add_argument('--optim',type=str,default='SGD')
parser.add_argument('--weight_decay',type=float,default=0)
parser.add_argument('--net_select',type=str,default='elec')



args=parser.parse_args()





