import hashlib
import unittest

from pairception.upstream import COMMIT, ROOT

HASHES = {
    'README.md': '92d066190124bec58d9d1ea156f837d86b8125fe0ad00b6695d5593da39e990d',
    'config/config_cfg.py': '2fdce849cacdc122fa63ee04385e3bc9d212139a04b44188c15d58bd99cc68b5',
    'datasets/__init__.py': '8aba8fcdad5e0175527a492824cfd968f14a609e8edd16ca180adea59f8c5300',
    'datasets/coco.py': '584cc28bc21434f84c10dd3926f8990b342647f86ac157ff63e8cd004156ccbc',
    'datasets/coco_eval.py': 'a80b849f2af5221fa79c391fa7b2ae1ae016a68c0a6c664b5c7d7f0e030a64e8',
    'datasets/coco_panoptic.py': '589b5eb175ad0e713ac3548314f653c96fc2e0be93f36d0f10a9d1f85c4bf145',
    'datasets/cocogrounding_eval.py': '7107dc8c308c08f9eaaf4dd5e24fafd07751145e33a71b6a073092ba8b65d6b8',
    'datasets/data_util.py': '5ce0f467c09e5d333bb4f2f1ad76b5b8f09c0604ceb9b1743f4ac8738f13fb8b',
    'datasets/dataset.py': '282f5b4b8880450c58a968fd60b3f8024350bc429adf6485884548ec20efe9d6',
    'datasets/odvg.py': 'c1e1b99d471cacc165d03590050be6afc90d57a6334e311242b7b839c0b51f15',
    'datasets/panoptic_eval.py': 'a8d7117160f57f25d35270d50a17be84dcfa75d3b4903a13bc1f41aeebe05b0c',
    'datasets/random_crop.py': '5ee8b7d1752ad10e6d22603c739ea6599a77aea3228afb5ebae489f6005e82d6',
    'datasets/sltransform.py': '464485f08fe3524fcce41eed248e7676374a27c4092f5bb674c140b43b477d45',
    'datasets/transforms.py': 'c7652cec14485fcc83be3c843935f34f3a5fd9e7e74347bce600466b323bf1ee',
    'demo.py': 'e7d31287ad32e0e6427627c1088b707dafd24f4f838e79a0bea8bd2dfc784638',
    'engine.py': '6714b668f7334ae1fdacb1877a49e81e1f104ea3bc788ddaf6728ef7174e4923',
    'eval.py': '9437684a1cec3644517a1b726cfa7b6b0280cdc4d29e57bce894aaeb2cd5ece5',
    'eval.sh': '74de86527cba72c9abc577ca486d0a11d15c51eadd819032408adf4d1d8aaf67',
    'model/AerialVG/__init__.py': '583ffac4e3dc5bc41fd0c6dec36d37ee1d61da91640b8c7c85aed3ccf9c51519',
    'model/AerialVG/aerialvg.py': '9f1884d0da90a64aad992ade134bb93dc40823a9d196a8c805028910888195f1',
    'model/AerialVG/backbone/__init__.py': 'a324d241d6a8d4a9d4484689c531b790c44abeb56076901058a2541860ce2d2f',
    'model/AerialVG/backbone/backbone.py': 'fe494cb68abb6207ce9b5963103262f34055a601c2e92793e39c14610c1f64ec',
    'model/AerialVG/backbone/position_encoding.py': '08e490186e818a75dc95c7f16fea8a1b9fcf66c288722b05422606882b57070e',
    'model/AerialVG/backbone/swin_transformer.py': '26e71d9a4519d513af777f034def8d09a036b3c83b85a39595ac23f93882170b',
    'model/AerialVG/bertwarper.py': 'd72d92064ff6028949d7823c7571e2b21040ad1574fddd4d25c86d3494ebb49f',
    'model/AerialVG/csrc/MsDeformAttn/ms_deform_attn.h': '88729d7ee3f72b03dd9738500d9f669c4f207b2f3bcbbea9a6adc3e75efe4e54',
    'model/AerialVG/csrc/MsDeformAttn/ms_deform_attn_cpu.cpp': 'dc6f6553f6944013106dd5a32de8a92f1f63deb26c12d295a6d29c373a3a99cc',
    'model/AerialVG/csrc/MsDeformAttn/ms_deform_attn_cpu.h': '624f50c52e85e92b7fd0d9ecb88eb44a0f6007add2ce375c78c336daa8a22d35',
    'model/AerialVG/csrc/MsDeformAttn/ms_deform_attn_cuda.cu': 'c614c8c68d2d710b3caf0c800ccf37c9534ecc87d4f0e758974cba3148e841b1',
    'model/AerialVG/csrc/MsDeformAttn/ms_deform_attn_cuda.h': 'c3bb1da1377c3005e8eb832e85b5a15db55548ddd8b1683cef14e0b95dba401a',
    'model/AerialVG/csrc/MsDeformAttn/ms_deform_im2col_cuda.cuh': '136729520587cbf0593209d692af61a1e8af5909bcf34b8bacb8844878496bad',
    'model/AerialVG/csrc/cuda_version.cu': 'bea86b60bd6afb698edbf49ce4671f4e0d33897fa3fd03677397b8c21601c549',
    'model/AerialVG/csrc/vision.cpp': '7b3f0c5a952e52196a177c9cdb423e11c5fa2724b22f6e0b31b55066ea102912',
    'model/AerialVG/fuse_modules.py': '4e65d518bfb4730c913e1fb7bdbfafc0f82e0cea38eeead0f65600530f7926c3',
    'model/AerialVG/ms_deform_attn.py': '788db2384dec6702aace3c57edcdcd6f5e743556bec7e7b0c8245dcb37c26f8d',
    'model/AerialVG/ops/functions/__init__.py': 'e1a39c1cf23562b11d696d81f95cf8bef368c48149b681d1eadc7f6d188edd3a',
    'model/AerialVG/ops/functions/ms_deform_attn_func.py': 'dd9906037ce7dc45b9ef6f39d4934bb4dddefc3e09049024496827714b6738bc',
    'model/AerialVG/ops/modules/__init__.py': '3a40cdcd146fc93c3f4ba9702924582f6502e5fcd1ceccf687e9d0e6ff2d7b5f',
    'model/AerialVG/ops/modules/ms_deform_attn.py': '658e8180b715775b3f8d3fb9c25932ca4bc618bc94a74439ed0eecace4857156',
    'model/AerialVG/ops/setup.py': '9675fc02e522204d6f3af59dca6dcad00b59a8b2b192e30e54f3e04279e279f2',
    'model/AerialVG/ops/src/cpu/ms_deform_attn_cpu.cpp': 'd7f7c3d2df2484f3905005e13c8dc12dd943fd4451a59a059bda367eefa4dccd',
    'model/AerialVG/ops/src/cpu/ms_deform_attn_cpu.h': 'b9114ab52ea7610184b3eb9672731c3d1a7ce92e69bfdec6cf661ae988416aef',
    'model/AerialVG/ops/src/cuda/ms_deform_attn_cuda.cu': '65cf318c67d99d6b163027e7190bc8776ff16a7197b332ad537f70cf6cc12be0',
    'model/AerialVG/ops/src/cuda/ms_deform_attn_cuda.h': 'c9cd9345e0881dfeebbfa5e98b6e6990a3d415a430bcbaaba3706ce5a2379401',
    'model/AerialVG/ops/src/cuda/ms_deform_im2col_cuda.cuh': '136729520587cbf0593209d692af61a1e8af5909bcf34b8bacb8844878496bad',
    'model/AerialVG/ops/src/ms_deform_attn.h': '2fdc244f99e3e3b9af8dae02e09139de3f546eef83748495c3f175f48278cf41',
    'model/AerialVG/ops/src/vision.cpp': '8dec032236873cee3ae86921dabbbb02c62ffd95ddbb83aef0b1b856771e8a61',
    'model/AerialVG/ops/test.py': '4d23eb5e4dc3a17d43d2191f775b9832e3128f48ec646b855cac41c71d963ee5',
    'model/AerialVG/relation.py': 'ea919c7c0bd8454041d69c4087d9837a13d8822dd52c7cd5100d2cd0e2365b6b',
    'model/AerialVG/transformer.py': '5ac454174dbbd662f557fc1a047e73497a037991ed2308bec469c273ed7cbc81',
    'model/AerialVG/transformer_vanilla.py': 'c4e69af49cd5070dd51aa9a4e33664e076ee98d3d08bb3960e6e8dceebec53d8',
    'model/AerialVG/utils.py': '2827ba921acaa263db7533afb6a818473a957fb2ef559c4f3407da72028a0842',
    'model/__init__.py': 'a948d186604b95613dd6dbc74e1d4f963819746bc946dfbe7ef882f10a3aac9a',
    'model/registry.py': '5b3e544d71250acfcf2137f886a6eed206e958d7493dfd962cdc5f2aa3e684aa',
    'requirements.txt': '4ca011488a0fbe3f39000c15640fc556ad126b83a572662e1dd388806061f0a2',
    'util/__init__.py': '9e535c45af09ac24ef8d5631a151f77febafe72a92ad6bfeb2d941f0c4a54e35',
    'util/box_ops.py': '9ca6f03a1df11fa7e1b9a6056e0f5a79b67682fd4c275de41353559daef7d327',
    'util/get_param_dicts.py': '1f091db7714ed7db7b1a26c078bc46374a98c29b0d56ebd268e3916962374901',
    'util/get_tokenlizer.py': '429109fac93130382040958edd3f79f26fe4c6f65b5f10659f3cbc40fbdf1ed9',
    'util/logger.py': '1455b7b5d68ac7c0ab3c0dd472a9aee1d7fa4903bc70aee84f33057c409edb7f',
    'util/misc.py': '821bf3b3a643a872981da5080b11bd35167c0c81ca3c34577f3b882ab6a866d5',
    'util/slconfig.py': 'af79fb9d2cbc969813873e257ffbad8890e23fcce9eab0c210b68b0a0279cf2f',
    'util/slio.py': 'fe8dd5e3efae4a93180c0f9319af838a4f22444e1a7544b496a7cd713bf5b461',
    'util/time_counter.py': '14e08e1416920094e760b0e6082fbe374bf7838b8a0712a755c071bc41c25dad',
    'util/utils.py': '37e03ba36cbdf50b1465f89c6d6b906aec80c2c594b497959f604aad8123e3c0',
    'util/visualizer.py': 'a8d6415a7eaf68ba08f3f81b93ae868550a049ea1ae23ef41c507e601a24d0f8',
    'util/vl_utils.py': '19ecd4beadcd2e9a1900e6f6b884040e1000a7783ac235d9e3138d0fac874150',
}


class UpstreamTest(unittest.TestCase):
    def test_files_are_unchanged(self):
        self.assertEqual(COMMIT, 'db33e1dc733911853c5e132394d2e00b024334db')
        files = sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        self.assertEqual(files, sorted(HASHES))
        for name, digest in HASHES.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
