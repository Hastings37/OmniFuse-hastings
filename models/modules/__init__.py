from .DiT_arch import DiT_L_2, DiT_L_4, DiT_B_2, DiT_B_4, DiT_S_4, DiT_S_8
from .UNet_arch import UNet
from .DenoisingUNet_arch import ConditionalUNet
from .DenoisingNAFNet_arch import ConditionalNAFNet, CNAFNetLocal
from .FusionNet_arch import FusionNet
from .Clip_arch import FrozenCLIPTextEmbedder, FrozenClipImageEmbedder
from .Clip_arch import FrozenCLIPTextEmbedder, FrozenClipImageEmbedder
from .mix_segnext import SegNeXt

from .ae_arch import TransformerUNet # 我们自己写的自编码器模型

# from .vae_arch import TransformerUNet # 这里引入的不能是同样的名字的 Unet 类；