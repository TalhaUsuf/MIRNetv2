import importlib
from os import path as osp

from basicsr.utils import scandir

# automatically scan and import arch modules
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder)
    if v.endswith('_arch.py')
]
# import all the arch modules
_arch_modules = [
    importlib.import_module(f'basicsr.models.archs.{file_name}')
    for file_name in arch_filenames
]


def dynamic_instantiation(modules, cls_type, opt):
    """Dynamically instantiate class.

    Args:
        modules (list[importlib modules]): List of modules from importlib
            files.
        cls_type (str): Class type.
        opt (dict): Class initialization kwargs.

    Returns:
        class: Instantiated class.
    """

    for module in modules: # module is basicsr/models/archs/mirnet_v2_arch.py
        cls_ = getattr(module, cls_type, None) # cls_type is MIRNet_v2
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_type} is not found.')
    return cls_(**opt)


# ============================================================
#   ➡ called by basicsr/models/image_restoration_model.py L68
# ============================================================
def define_network(opt):
    """
    network_g:
      type: MIRNet_v2
      inp_channels: 6
      out_channels: 3
      n_feat: 80
      chan_factor: 1.5
      n_RRG: 4
      n_MRB: 2
      height: 3
      width: 2
      scale: 1
      task: 'defocus_deblurring'
    """
    network_type = opt.pop('type')
    net = dynamic_instantiation(_arch_modules, network_type, opt)
    return net
