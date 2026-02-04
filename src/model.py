import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


class EdgeInteractionLayer(MessagePassing):
    """Graph interaction layer that combines node & edge features on edges.
    """
    def __init__(
        self, 
        n_in: int, 
        n_hidden: int, 
        n_latent: int, 
        aggr: list[str]=["sum", "max", "mean"], 
        act_fn: nn.Module=nn.SiLU
    ):
        super(EdgeInteractionLayer, self).__init__(aggr)

        self.mlp = nn.Sequential(
            nn.Linear(n_in, n_hidden, bias=True),
            nn.LayerNorm(n_hidden),
            act_fn(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.LayerNorm(n_hidden),
            act_fn(),
            nn.Linear(n_hidden, n_latent, bias=True),
        )

    def forward(self, x, edge_index, edge_attr, u):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, u=u)

    def message(self, x_i, x_j, edge_attr, u_i):
        inputs = torch.cat([x_i, x_j, edge_attr, u_i], dim=-1)
        return self.mlp(inputs)


class EdgeInteractionGNN(nn.Module):
    """Graph net over nodes and edges with multiple unshared layers, and sequential layers with residual connections.
    """
    def __init__(
        self, 
        n_layers: int=1, 
        node_features: int=6, 
        edge_features: int=1, 
        graph_features: int=2,
        hidden_channels: int=64, 
        aggr: list[str]=["sum", "max", "mean"], 
        latent_channels: int=64, 
        n_out: int=1, 
        n_unshared_layers: int=4, 
        act_fn: nn.Module=nn.SiLU
    ):
        super(EdgeInteractionGNN, self).__init__()

        self.n_in = node_features
        self.n_graph = graph_features
        self.n_hidden = hidden_channels
        self.n_out = n_out
        self.n_unshared = n_unshared_layers
        self.n_pool = (len(aggr) if isinstance(aggr, list) else 1) 
        
        layers = [
            nn.ModuleList([
                EdgeInteractionLayer(2 * node_features + edge_features + graph_features, hidden_channels, latent_channels, aggr=aggr, act_fn=act_fn)
                for _ in range(n_unshared_layers)
            ])
        ]
        for _ in range(n_layers - 1):
            layers += [
                nn.ModuleList([
                    EdgeInteractionLayer(
                        self.n_pool * (2 * latent_channels * self.n_unshared) + edge_features + graph_features, 
                        hidden_channels, 
                        latent_channels, 
                        aggr=aggr, 
                        act_fn=act_fn
                    ) for _ in range(self.n_unshared)
                ])
            ]
   
        self.layers = nn.ModuleList(layers)
        
        self.env2node_mlp = nn.Sequential(
            nn.Linear(self.n_pool * self.n_unshared * latent_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, latent_channels, bias=True)
        )

        if node_features > 0:
            self.node2node_mlp = nn.Sequential(
                nn.Linear(node_features, hidden_channels, bias=True),
                nn.LayerNorm(hidden_channels),
                act_fn(),
                nn.Linear(hidden_channels, hidden_channels, bias=True),
                nn.LayerNorm(hidden_channels),
                act_fn(),
                nn.Linear(hidden_channels, latent_channels, bias=True)
            )
            fc_input_dim = 2 * latent_channels + graph_features
        else:
            self.node2node_mlp = None
            fc_input_dim = latent_channels + graph_features

        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, 2*n_out, bias=True)
        )
    
    def forward(self, data: Data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch    
        u = data.u[batch]
        
        if u.dim() == 1:
            u = u.unsqueeze(-1)
    
        h = torch.cat([
            layer(x, edge_index, edge_attr, u)
            for layer in self.layers[0]
        ], dim=-1)
    
        for layer_group in self.layers[1:]:
            h = h + torch.cat([
                layer(h, edge_index, edge_attr, u)
                for layer in layer_group
            ], dim=-1)
    
        if self.node2node_mlp is not None:
            out = torch.cat([self.env2node_mlp(h), self.node2node_mlp(x), u], dim=-1)
        else:
            out = torch.cat([self.env2node_mlp(h), u], dim=-1)
        
        return self.fc(out)
