import torch.nn as nn
from collections import OrderedDict
import torch

class MLP(nn.Module):
    def __init__(self, layers, activation_function , p_dropout=0.1) -> None:
        super(MLP, self).__init__()

        self.layers = layers
        self.activation = activation_function
        

        
        layer_list = list()    

        for i in range(len(self.layers)-2):
            layer_list.append(
                ('layer_%d' % i, nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            layer_list.append(('dropout_%d' % i, nn.Dropout(p = p_dropout)))
        layer_list.append(('layer_%d' % (len(self.layers)-1), nn.Linear(self.layers[-2], self.layers[-1])))
        self.mlp = nn.Sequential(OrderedDict(layer_list))
        


    def forward(self, x ):
        output = self.mlp(x)
        return output
