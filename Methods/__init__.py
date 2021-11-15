
from dataclasses import dataclass
from simple_parsing import mutable_field
from .models.mntdp import MNTDP_net
# from .models.fixed_extractor import OvAInn
from .models.hat import HAT
from .models.cnn_independent_experts import ExpertMixture
from .models.LMC import LMC_net
from .models.LMC_components import LMC_conv_block
from .models.cnn_soft_gated_lifelong_dynamic import CNNSoftGatedLLDynamic

@dataclass
class ModelOptions():
    LMC: LMC_net.Options = mutable_field(LMC_net.Options)
    Module: LMC_conv_block.Options = mutable_field(LMC_conv_block.Options)
    Experts: ExpertMixture.Options = mutable_field(ExpertMixture.Options)
    MNTDP: MNTDP_net.Options = mutable_field(MNTDP_net.Options)
    HAT_Method: HAT.Options = mutable_field(HAT.Options)
    SGNet: CNNSoftGatedLLDynamic.Options = mutable_field(CNNSoftGatedLLDynamic.Options)