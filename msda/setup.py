from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

UPSTREAM = Path(__file__).resolve().parents[1] / 'external' / 'aerialvg' / 'model' / 'AerialVG' / 'ops' / 'src'
SOURCE = Path(__file__).resolve().parent / 'build' / 'src'
RENAMES = (('.type().is_cuda()', '.is_cuda()'), ('value.type(),', 'value.scalar_type(),'), ('.data<', '.data_ptr<'))

for path in sorted(UPSTREAM.rglob('*')):
    if path.is_file():
        text = path.read_text()
        for old, new in RENAMES:
            text = text.replace(old, new)
        target = SOURCE / path.relative_to(UPSTREAM)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

setup(
    name='MultiScaleDeformableAttention',
    version='1.0',
    ext_modules=[
        CUDAExtension(
            'MultiScaleDeformableAttention',
            sources=[str(path) for path in sorted([*SOURCE.glob('*.cpp'), *SOURCE.glob('cpu/*.cpp'),
                                                    *SOURCE.glob('cuda/*.cu')])],
            include_dirs=[str(SOURCE)],
            define_macros=[('WITH_CUDA', None)],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-DCUDA_HAS_FP16=1', '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__',
                         '-D__CUDA_NO_HALF2_OPERATORS__'],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
