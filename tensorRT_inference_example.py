import numpy as np
import time
import argparse
from typing import Any, Tuple, List
try:
    import tensorrt as trt
    import pycuda.autoinit
    import pycuda.driver as cuda
except ImportError:
    print("Failed to load tensorrt, pycuda")
    trt = None
    cuda = None

class TensorRTProcessor():
    """TensorRT model wrapper."""
    def __init__(self, model_path: str, batch: int) -> None:
        """
        Initializes the TRTWrapper with a path to the model and batch size.

        Args:
            model_path (str): Path to the .trt model file.
            batch (int): Batch size for the model.
        """
        self.model_path = model_path
        self._batch = batch

        self._bindings = None

    @property
    def batch(self) -> int:
        """Gets the batch size."""
        return self._batch

    @batch.setter
    def batch(self, value: int) -> None:
        """Sets the batch size."""
        self._batch = value

    @property
    def bindings(self) -> Any:
        """Gets the bindings."""
        return self._bindings

    @bindings.setter
    def bindings(self, value: Any) -> None:
        """Sets the bindings."""
        self._bindings = value

    def load_model(self) -> None:
        """
        Load a serialized TensorRT engine from a .trt file, 
        create a new execution context for this engine.
        """
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        # serialized ICudEngine을 deserializ
        runtime = trt.Runtime(TRT_LOGGER) 
        # plugin 사용
        trt.init_libnvinfer_plugins(None, "")
        with open(self.model_path, 'rb') as f:
            # trt을 불러온 뒤 serialized ICudEngine을 deserialize
            self.engine = runtime.deserialize_cuda_engine(f.read()) 
        # ICudEngine inference를 위한 context 생성
        self.context = self.engine.create_execution_context()
        # assert self.engine 
        assert self.context
        
        self.allocate_buffer()

    def allocate_buffer(self) -> None:
        """
        Allocate memory on the GPU and host based on the engine bindings. 
        It sets up the appropriate bindings for input and output.
        """
        # I/O 바인딩(bindings) 설정
        self.inputs = []
        self.outputs = []
        self.allocations = []

        for i in range(self.engine.num_bindings):
            is_input = False
            # i번째 binding이 input인지 확인
            if self.engine.binding_is_input(i):
                is_input = True 
            # i번째 binding의 name
            name = self.engine.get_binding_name(i)
            # i번째 binding의 data type
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            # i번째 binding의 shape
            shape = self.context.get_binding_shape(i)

            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3 # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            # data type의 bit수
            size = dtype.itemsize
            for s in shape:
                """
                data type * 각 shape element 을 곱하여 size에 할당
                (e.g input의 경우 [B, H, W, C])
                """
                size *= s 

            allocation = cuda.mem_alloc(size) # 해당 size만큼의 GPU memory allocation함
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            # binding이 input인 경우
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            # binding은 모두 output임
            else: 
                self.outputs.append(binding)
            print("{} '{}' with shape {} and dtype {}".format(
                "Input" if is_input else "Output",
                binding['name'], binding['shape'], binding['dtype']))        

        # 검증
        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def inference(self, image: np.ndarray) -> np.ndarray:
        """
        Run the inference engine with the provided image as input.

        Args:
            image (np.ndarray): The input image as a numpy array.

        Returns:
            np.ndarray: The output from the inference engine.
        """
        image = image.astype(np.float32)
        image = np.ascontiguousarray(image) 
        # input image array(from host) -> GPU(device)로 전달
        cuda.memcpy_htod(self.inputs[0]['allocation'], image) 
        # inference
        self.context.execute_v2(self.allocations) 
        for i in range(len(self.outputs)):
            # GPU(device) -> Host 전달
            cuda.memcpy_dtoh(self.outputs[i]['host_allocation'], self.outputs[i]['allocation'])
         # 출력 결과, 복수 출력인 경우 [1], [2] index로 접근
        outputs = self.outputs[0]['host_allocation']

        result = outputs
        return result

    def input_spec(self) -> Tuple:
        """
        Returns the shape and datatype of the input tensor.

        Returns:
            tuple: Shape and datatype of input tensor.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self) -> List[Tuple]:
        """
        Returns the shape and datatype of the output tensors.

        Returns:
            list of tuple: List of shape and datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

def create_model_wrapper(model_path: str, batch_size: int):
    """Create model wrapper class."""
    assert trt and cuda, f"Loading TensorRT, Pycuda lib failed."
    model_wrapper = TensorRTProcessor(model_path, batch_size)

    return model_wrapper

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.trt', help='TensorRT .trt, .engine 경로')
    parser.add_argument('--b', type=int, default=1, help='배치 사이즈 크기 설정')
    parser.add_argument('--h', type=int, default=36, help='입력 높이 해상도 설정')
    parser.add_argument('--w', type=int, default=36, help='입력 너비 해상도 설정')
    parser.add_argument('--c', type=int, default=3, help='입력 채널 해상도 설정')
    parser.add_argument('--torch_mode', type=bool, default=False, action="store_true", help='파이토치 모델 shape B,C,H,W')


    args = parser.parse_args()

    # load model 
    model_wrapper = create_model_wrapper(
        model_path=args.model,
        batch_size=args.b,
    )

    model_wrapper.load_model()

    if args.torch_mode:
        input_shape = [args.b, args.c, args.h, args.w]
    else:
        input_shape = [args.b, args.h, args.w, args.c]

    dummy = np.ones().astype(np.float32)
    result = model_wrapper.inference(dummy)

    for i in range(1000):
        start = time.time()
        result = model_wrapper.inference(dummy)
        # print(result)
        stop = time.time()
        print('predict duration {:.9}s'.format(stop - start))