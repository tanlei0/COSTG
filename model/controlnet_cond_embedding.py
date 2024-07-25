from torch import nn
from typing import Tuple
from diffusers.models.controlnet import zero_module
from torch.nn import functional as F

class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16,32,64,128,256,512,1024),
        UNet_in_channels: Tuple[int] = (320, 640, 1280, 1280)
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        self.UNet_in_chs = UNet_in_channels

        self.ms_blocks = nn.ModuleList([])

        self.ms_idx = block_out_channels[-len(UNet_in_channels):]

        for i in range(len(block_out_channels)-1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
            if i >= len(block_out_channels) - len(self.UNet_in_chs) -1:
                self.ms_blocks.append(zero_module(nn.Conv2d(channel_out, self.UNet_in_chs[i - (len(block_out_channels) - len(self.UNet_in_chs))+1], kernel_size=3, padding=1)))



    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        ms_embeddings = []
        j = 0
        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
            if embedding.shape[1] == self.ms_idx[j]:
                ms_embedding = self.ms_blocks[j](embedding)
                ms_embeddings.append(ms_embedding)
                j = j + 1

        return ms_embeddings
