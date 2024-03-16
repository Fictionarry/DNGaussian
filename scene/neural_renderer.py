import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, p=0.1, training=self.training)
                
        return x


class GridRenderer(nn.Module):
    def __init__(self,
                 bound = 1.,
                 coord_center=[0., 0., 0.],
                 keep_sigma=False
                 ):
        super().__init__()
        self.register_buffer('bound', torch.as_tensor(bound, dtype=torch.float32).detach())
        self.register_buffer('coord_center', torch.as_tensor(coord_center, dtype=torch.float32).detach())

        self.keep_sigma = keep_sigma
        self.sigma_results_static = None

        self.num_levels = 16
        self.level_dim = 2
        self.base_resolution = 16
        self.table_size = 19
        self.desired_resolution = 512
        self.encoder_x, self.in_dim_x = self.create_encoder()

        ## sigma network
        self.num_layers = 3
        self.hidden_dim = 64
        self.geo_feat_dim = 64
        self.sigma_net = MLP(self.in_dim_x, 1 + self.geo_feat_dim, self.hidden_dim, self.num_layers)
        ## color network
        self.num_layers_color = 2
        self.hidden_dim_color = 64
        self.encoder_dir, self.in_dim_dir = get_encoder('sphere_harmonics')
        self.color_net = MLP(self.in_dim_dir + self.geo_feat_dim, 3, self.hidden_dim_color, self.num_layers_color)

    
    def create_encoder(self):
        self.encoder_x, self.in_dim_x = get_encoder(
            'hashgrid', input_dim=3, num_levels=self.num_levels, level_dim=self.level_dim, 
            base_resolution=self.base_resolution, log2_hashmap_size=self.table_size, desired_resolution=self.desired_resolution * self.bound.cpu())
        return self.encoder_x, self.in_dim_x

    def recover_from_ckpt(self, state_dict):
        self.bound = state_dict['bound']
        self.encoder_x, self.in_dim_x = self.create_encoder()
        self.load_state_dict(state_dict)

    def encode_x(self, x):
        # x: [N, 3], in [-bound, bound]
        return self.encoder_x(x - self.coord_center, bound=self.bound)


    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        enc_x = self.encode_x(x)
        sigma_result = self.density(x, enc_x)
        sigma = sigma_result['sigma']
        color = self.color(sigma_result, d)
        return sigma, color
    

    def color(self, sigma_result, d):
        geo_feat = sigma_result['geo_feat']
        enc_d = self.encoder_dir(d)
        h = torch.cat([enc_d, geo_feat], dim=-1)

        h_color = self.color_net(h)
        color = torch.sigmoid(h_color)*(1 + 2*0.001) - 0.001
        return color


    def density(self, x, enc_x=None):
        # x: [N, 3], in [-bound, bound]
        if self.keep_sigma and self.sigma_results_static is not None:
            return self.sigma_results_static
        
        if enc_x is None:
            enc_x = self.encode_x(x)

        h = self.sigma_net(enc_x)
        sigma = h[..., 0]
        # sigma = torch.exp(h[..., 0])
        # sigma = torch.sigmoid(h[..., 0])
        geo_feat = h[..., 1:]

        if self.keep_sigma:
            self.sigma_results_static = {
                    'sigma': sigma,
                    'geo_feat': geo_feat,
                }

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.encoder_x.parameters(), 'name': 'neural_encoder', 'lr': lr},
            {'params': self.sigma_net.parameters(), 'name': 'neural_sigma', 'lr': lr_net, 'weight_decay': wd},
            {'params': self.color_net.parameters(), 'name': 'neural_color', 'lr': lr_net, 'weight_decay': wd}, 
        ]
        
        return params