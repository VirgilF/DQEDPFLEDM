import os
from set import args
from Server import Server
from data_spilt import loader,test
import torch
import numpy as np
import time



if __name__=='__main__':
    torch.manual_seed(1)
    np.random.seed(1)
    args.device=torch.device('cpu')
    print(args.device)
    st=time.time()
    client_data = loader()
    test_loader = test()
    server=Server(args,client_data,test_loader)
    server.set_FL()
    result=server.fit()
    et=time.time()
    print('--------------time---------------',et-st)
    torch.save(result,r'C:\Users\fcy\Desktop\FL+sv+dp\FL+sv+dp\result\mse')


