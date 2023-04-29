from torch import nn
from heuds.utils import get_activation_nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, activation = 'relu', dropout=0.1):
        super(MLP, self).__init__()
        activation = get_activation_nn(activation)
        dropout = nn.Dropout(p = dropout)
        layer_list = [nn.Linear(input_dim, hidden_dim),
                                      activation(),
                                      dropout]
        for _ in range(n_layers - 1):
            layer_list += [nn.Linear(hidden_dim, hidden_dim),
                                      activation(),
                                      dropout]
        layer_list += [nn.Linear(hidden_dim, output_dim)]

        self.model = nn.Sequential(*layer_list)

    def forward(self, input):
        out = self.model(input)
        return out
