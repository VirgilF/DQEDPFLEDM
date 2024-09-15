import matplotlib.pyplot as plt
import torch
import numpy as np


mse=torch.load(r'./mse')
data=np.array(mse['loss'])
print(data[0])
y=np.arange(len(data))
plt.plot(y, data, linestyle='-')
plt.ylim(0,100)
plt.xlabel('epoch')
plt.ylabel('mse')
plt.savefig('./mse.png')
plt.show()
