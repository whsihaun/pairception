import importlib.util
import os
import sys
import warnings
from pathlib import Path

import transformers

ROOT = Path(__file__).resolve().parents[1] / 'external' / 'aerialvg'
COMMIT = 'db33e1dc733911853c5e132394d2e00b024334db'

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
for message in ('Importing from timm.models.layers is deprecated', r'`torch.cuda.amp.autocast\(args...\)` is deprecated',
                'torch.meshgrid: in an upcoming release', 'Failed to load custom C\\+\\+ ops',
                'invalid escape sequence'):
    warnings.filterwarnings('ignore', message=message)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model.AerialVG import aerialvg, backbone, bertwarper, ms_deform_attn, transformer, utils  # noqa: E402
from util import box_ops, misc  # noqa: E402


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transforms = _load('aerialvg_transforms', 'datasets/transforms.py')


def require_kernel():
    if not hasattr(ms_deform_attn, '_C'):
        raise RuntimeError('The fused deformable attention kernel is not installed; run the first line of '
                           'scripts/prepare.sh')
