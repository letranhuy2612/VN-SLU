from transformers import Wav2Vec2Processor
import onnxruntime as rt
import soundfile as sf
import numpy as np
import scipy.signal

ONNX_PATH = "checkpoints/wav2vec.onnx"

processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")

sess_options = rt.SessionOptions()
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
session = rt.InferenceSession(ONNX_PATH, sess_options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def predict(file):
  speech_array, sr = sf.read(file)
  speech_array = scipy.signal.resample(speech_array, int(float(speech_array.shape[0])/sr*16000))
  features = processor(speech_array, sampling_rate=16000, return_tensors="pt")
  input_values = features.input_values
  print(input_values)
  onnx_outputs = session.run(None, {session.get_inputs()[0].name: input_values.numpy()})[0]
  prediction = np.argmax(onnx_outputs, axis=-1)
  return processor.decode(prediction.squeeze().tolist())
print(predict('64a00b509e6641dda2b4535e.wav'))