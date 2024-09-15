import copy
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import warnings
from opacus import PrivacyEngine
warnings.filterwarnings("ignore")



class Client():
    def __init__(self,args,id,local_data,device):
        self.id=id
        self.local_data=local_data
        self.device=args.device
        self.args=args


    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self,model):
        self.__model=model

    def setup(self,epoch,lossfun,optim):
        self.epoch=epoch
        self.loss_f=lossfun
        self.optim=optim


    def Fedavg_update(self):
        self.model.train()
        self.model.to(self.device)
        optim=self.optim(self.model.parameters(),lr=self.args.lr, weight_decay=self.args.weight_decay,momentum=0.9)
        privacy_engine = PrivacyEngine(self.model, batch_size=128, sample_size=10000, alphas=range(2, 32),
                                       noise_multiplier=1.3, max_grad_norm=1.0)
        privacy_engine.attach(optim)

        for e in tqdm(range(self.epoch),desc='--client {} is updating'.format(self.id)):
            for data1, labels in self.local_data:
                data1,labels = data1.float().to(self.device), labels.float().to(self.device)
                optim.zero_grad()
                outputs = self.model(data1)
                loss = self.loss_f(outputs, labels)
                loss.backward()
                optim.step()
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")



    def eval(self,loader):
        self.model.eval()
        self.model.to(self.device)
        test_loss = 0

        with torch.no_grad():
            for data1, labels in loader:

                data1, labels = data1.float().to(self.device), labels.long().to(
                    self.device)
                outputs = self.model(data1)
                test_loss += self.loss_f(outputs, labels).item()
                torch.cuda.empty_cache()
        self.model.to("cpu")
        test_loss = test_loss / len(self.local_data)
        return test_loss