import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=3072, hidden_dim=1024, num_classes=10, num_hidden_layers=4):
        super(MLP, self).__init__()
        self.name = 'mlp'
        self.first_layer = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(input_size, hidden_dim),
                    nn.ReLU(),
        )
        
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_hidden_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, num_classes, bias=False)
        self.network = nn.Sequential(
            self.first_layer,
            *self.hidden_layers,
            self.output_layer
        )
    
    def forward(self, x):
        return self.network(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)