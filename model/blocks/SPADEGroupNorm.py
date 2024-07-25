import torch.nn as nn

class SPADEGroupNorm(nn.Module):
    def __init__(self, in_channels, out_channels, n_hidden,
                 eps = 1e-5, groups = 32):
        super().__init__()

        self.norm = nn.GroupNorm(groups, out_channels, affine=False) 

        self.eps = eps

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(in_channels, n_hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.mlp_gamma = nn.Conv2d(n_hidden, out_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(n_hidden, out_channels, kernel_size=3, padding=1)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        x = self.norm(x)
        
        # Part 2. produce scaling and bias conditioned on semantic map
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        if x.shape != segmap.shape:
            raise "Error: x.shape {} != segmap.shape {}".format(x.shape, segmap.shape)
        
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        return x * (1 + gamma) + beta