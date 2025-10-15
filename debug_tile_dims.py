import tvm
from tvm.script import tir as T
import tvm.script
import sys
import os

# Add passes directory to path
sys.path.append(os.path.join(os.path.dirname('.'), 'tilelang/tenstorrent/passes'))
from attach_tensor_accessor_tt import AttachTensorAccessorTT

@tvm.script.ir_module
class TestModule:
    @T.prim_func
    def func(A: T.Buffer((512, 384), 'float16')):
        T.evaluate(0)

func = TestModule['func']
func = func.with_attr('tt.buffer.A', {
    'memory': 'DRAM',
    'layout': 'interleaved',
    'tile_shape': [32, 32],
    'dtype': 'bf16'
})
TestModule['func'] = func

pass_a3 = AttachTensorAccessorTT()
result = pass_a3(TestModule)
func = result['func']

accessor_a = func.attrs['tt.tensor_accessor.A']
print('tile_dims type:', type(accessor_a['tile_dims']))
print('tile_dims value:', accessor_a['tile_dims'])
print('tile_dims repr:', repr(accessor_a['tile_dims']))
print('Comparison:', accessor_a['tile_dims'] == [32, 32])
if hasattr(accessor_a['tile_dims'], '__iter__'):
    print('List conversion:', list(accessor_a['tile_dims']))
    print('List comparison:', list(accessor_a['tile_dims']) == [32, 32])