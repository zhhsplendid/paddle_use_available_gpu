import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core

np.random.seed(123)

ONE_G = 2**30

# float32 takes 4 bytes. So 2.5 * ONE_G float32 constant will take 10 GiB
x = fluid.layers.fill_constant(shape=[1, 2.5 * ONE_G],
                               value=1,
                               dtype='float32')
y = fluid.layers.reduce_max(x, dim=1)

exe = fluid.Executor(core.CUDAPlace(0))
output = exe.run(program=fluid.default_main_program(), fetch_list=[y])

print("output = %f" % output[0])
print("Create data on GPU successfully")
