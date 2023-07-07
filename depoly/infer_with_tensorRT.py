'''
Date: 2023-07-07 17:37:42
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-07-07 19:07:32
FilePath: /QC-wrist/depoly/infer_with_tensorRT.py
Description: 
'''
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

# 1. 确定batch size大小，与导出的trt模型保持一致
BATCH_SIZE = 1          

# 2. 选择是否采用FP16精度，与导出的trt模型保持一致
USE_FP16 = False                                         
target_dtype = np.float16 if USE_FP16 else np.float32   
# 3. 创建Runtime，加载TRT引擎
f = open("./trt_model.trt", "rb")                     # 读取trt模型
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))   # 创建一个Runtime(传入记录器Logger)
engine = runtime.deserialize_cuda_engine(f.read())      # 从文件中加载trt引擎
context = engine.create_execution_context()             # 创建context

# 4. 分配input和output内存
input_batch = np.random.randn(BATCH_SIZE, 3, 960, 1920).astype(target_dtype)
output = np.empty([BATCH_SIZE, 1, 2], dtype = target_dtype)
d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()

# 5. 创建predict函数
def predict(batch): # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)  # 此处采用异步推理。如果想要同步推理，需将execute_async_v2替换成execute_v2
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()

    return output

# 6. 调用predict函数进行推理，并记录推理时间
preprocessed_inputs = np.array([input for input in input_batch])  # (BATCH_SIZE,224,224,3)——>(BATCH_SIZE,3,224,224)

print("Warming up...")
pred = predict(preprocessed_inputs)
print("Done warming up!")

t0 = time.time()
pred = predict(preprocessed_inputs)
t = time.time() - t0
print("Prediction cost {:.2f}ms".format(t*1000))