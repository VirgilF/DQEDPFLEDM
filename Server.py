import copy
import numpy as np
from collections import OrderedDict
from client import *
import torch
import torch.utils.data
# from model import MLP,LSTMModel
from model import *
import itertools


class Server():

     def __init__(self,args,train_loader_list,test_loader):
          self.args=args
          self.device='cpu'
          self.clients=None
          self._round=0
          if args.net_select=='elec':
               self.model=MLP()
          self.fraction=args.fraction
          self.num_round=args.global_epoch
          self.local_epoch=args.local_epoch
          self.num_clients=args.user_num
          self.seed=args.seed
          self.train_loader_list=train_loader_list
          self.test_loader=test_loader
          self.pre_model=[]
          if self.args.optim == 'SGD':
               self.optim = torch.optim.SGD
          if self.args.optim == 'Adam':
               self.optim = torch.optim.Adam
          if self.args.LOSS == 'CE':
               self.loss_f = torch.nn.CrossEntropyLoss()
          if self.args.LOSS == 'MSE':
               self.loss_f = torch.nn.MSELoss(reduction='mean')

     def set_FL(self):

          assert self._round==0
          torch.manual_seed(self.seed)
          # init_net(self.model,self.args)
          print(f"[Round: {str(self._round)}] --successfully initialized model\n")
          self.clients=self.create_clients()

          self.setup_clients(epoch=self.local_epoch,lossfun=self.loss_f,
                             optim=self.optim)
          self.send_model(None)

     def create_clients(self):
          clients_list=[]
          self.clients_attacked_list={}
          for i,loader in enumerate(self.train_loader_list):
               client=Client(self.args,id=i,local_data=loader,device=self.device)
               clients_list.append(client)
          # print(self.clients_attacked_list)
          print('--successfully create all {} clients'.format(self.num_clients))
          return clients_list


     def setup_clients(self,epoch,lossfun,optim):
          for i,client in enumerate(self.clients):
               client.setup(epoch,lossfun,optim)
          print('--successfully set up all {} clients'.format(self.num_clients))

     def send_model(self,random_select_idx):
          if random_select_idx is None:
               assert (self._round == 0) or (self._round == self.num_round)
               for i,client in enumerate(self.clients):
                    client.model=copy.deepcopy(self.model)
               print('--successfully transmitted models to all {} clients'.format(self.num_clients))
          else:
               assert self._round != 0
               for i,idx in enumerate(random_select_idx):
                    if self.args.algorithm == 'Fedcon':
                         self.pre_model.append(self.clients[idx].model)
                    self.clients[idx].model=copy.deepcopy(self.model)
               print('--successfully transmitted models to SELECTED {} clients'.format(len(random_select_idx)))
               print('    ')

     def select_client(self):

          num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
          print('--random select {} clients'.format(num_sampled_clients))
          random_select_idx = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients,
                                             replace=False).tolist())
          return random_select_idx

     def clients_update(self,random_select_idx,c_epoch):
          if self.args.algorithm=='Fedavg':
               for idx in random_select_idx:
                    self.clients[idx].Fedavg_update()
          print('--The selected {} clients are updated '.format(len(random_select_idx)))

     def calculate_sv(self,models, args):

          def averaging_func(models, args):
               w_avg = OrderedDict()
               for it, (idx,_) in enumerate(models.items()):
                    local_weights = models[idx].state_dict()
                    for key in self.model.state_dict().keys():
                         if it == 0:
                              w_avg[key] = 1/len(models) * local_weights[key]
                         else:
                              w_avg[key] += 1/len(models) * local_weights[key]

               list(models.items())[0][1].load_state_dict(w_avg)

               return list(models.items())[0][1]

          def global_eval(model):
               model.eval()
               loss_f = torch.nn.MSELoss(reduction='mean')
               test_loss, correct, len_data = 0, 0, 0
               prob_all = []
               label_all = []
               with torch.no_grad():
                    for data1, labels in self.test_loader:
                         data1, labels = data1.float(), labels.long()
                         outputs = model(data1)
                         test_loss += loss_f(outputs, labels).item()
               self.model.to("cpu")
               test_loss = test_loss / len(self.test_loader)
               return -test_loss
          all_perms = list(itertools.permutations(list(range(args.user_num))))
          marginal_contributions = []
          history = {}
          print('----------------------Evaluating SV...---------------------------')
          for perm in all_perms:
               perm_values = {}
               local_models = {}

               for client_id in perm:
                    model = copy.deepcopy(models[client_id])
                    local_models[client_id] = model

                    if len(perm_values.keys()) == 0:
                         index = (client_id,)
                    else:
                         index = tuple(sorted(list(tuple(perm_values.keys()) + (client_id,))))

                    if index in history.keys():
                         current_value = history[index]
                    else:
                         model = averaging_func(local_models,args)
                         current_value = global_eval(model)

                         history[index] = current_value

                    perm_values[client_id] = current_value - sum(perm_values.values())

               marginal_contributions.append(perm_values)
               print(perm, ':', marginal_contributions[-1])

          sv = {client_id: 0 for client_id,_ in enumerate(models)}

          # sum the marginal contributions
          for perm in marginal_contributions:
               for key, value in perm.items():
                    sv[key] += value

          # compute the average marginal contribution
          sv = {key: value / len(marginal_contributions) for key, value in sv.items()}
          print('----------------------Finsh Evaluating...---------------------------')
          print(sv)

          return sv


     def avg_model(self,random_select_idx,coefficients):
          model_list=[]
          for it, idx in enumerate(random_select_idx):
               model_list.append(copy.deepcopy(self.clients[idx].model))
          w_avg=OrderedDict()
          sv=self.calculate_sv(model_list,self.args)
          max_val = max(sv.values())
          min_val = min(sv.values())
          normalized_data = {}
          total = 0
          for key, value in sv.items():
               normalized_value = ((value - min_val) / (max_val - min_val))
               total += normalized_value
               normalized_data[key] = normalized_value
          for key in normalized_data:
               normalized_data[key] /= total
          normalized_data=list(normalized_data.values())
          for it, idx in enumerate(random_select_idx):
               local_weights = self.clients[idx].model.state_dict()
               for key in self.model.state_dict().keys():
                    if it == 0:
                         w_avg[key] = normalized_data[it] * local_weights[key]
                    else:

                         w_avg[key] += normalized_data[it] * local_weights[key]
          self.model.load_state_dict(w_avg)
          print('--updated weights of {} clients are successfully averaged!'.format(len(random_select_idx)))

     def clients_eval(self,random_select_idx):
          print('Evaluate selected {} client:'.format(len(random_select_idx)))
          for i,idx in enumerate(random_select_idx):
               self.clients[idx].eval()
          print('--Finished eval clients')

     def global_eval(self):
          self.model.eval()
          self.model.to(self.device)
          test_loss=0
          with torch.no_grad():
               for data1,labels in self.test_loader:
                    data1, labels = data1.float().to(self.device), labels.long().to(self.device)
                    outputs = self.model(data1)
                    test_loss += self.loss_f(outputs, labels).item()
                    if self.device == "cuda": torch.cuda.empty_cache()
          self.model.to("cpu")
          test_loss = test_loss / len(self.test_loader)
          return test_loss

     def train_Fed(self,c_epoch):
          random_select_idx=self.select_client()
          self.send_model(random_select_idx)
          self.clients_update(random_select_idx,c_epoch)
          coefficients = [1 / len(random_select_idx)]*len(random_select_idx)
          self.avg_model(random_select_idx,coefficients)

     def fit(self):
          print(f'[{str(self.args.algorithm)}]\n')
          self.result = {'loss': [], 'acc': []}
          loss_raw=self.global_eval()
          self.result['loss'].append(loss_raw)
          print(f'---[Round: {str(self._round)}]:raw_MSE=', self.result['loss'])
          for rd in range(self.num_round):
               self._round=rd+1
               print('-------------------------------------------------------------')
               print(f"[Round: {str(self._round)}] --successfully initialized model\n")
               self.train_Fed(rd)
               loss=self.global_eval()
               self.result['loss'].append(loss)
               print(f'---[Round: {str(self._round)}]:MSE=',self.result['loss'][-1])
          self.send_model(None)
          return self.result

















