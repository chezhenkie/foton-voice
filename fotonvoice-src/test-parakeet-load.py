import time, sys, onnxruntime as ort
D = r"C:\Users\lap1user\AppData\Local\fotonvoice-engine\models\parakeet\tdt-0.6b-v3"
opts = ort.SessionOptions()
for name, lvl in [("nemo128.onnx", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
                  ("encoder-model.int8.onnx", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
                  ("decoder_joint-model.int8.onnx", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)]:
    t0 = __import__("time").time()
    print(f"START {name} t={t0:.1f}", flush=True)
    try:
        s = ort.InferenceSession(str(D + "\\" + name), opts, providers=["CPUExecutionProvider"])
        print(f"DONE {name} in {__import__('time').time()-t0:.1f}s inputs={[ (i.name,i.shape) for i in s.get_inputs()]}", flush=True)
    except Exception as e:
        print(f"ERROR {name}: {e}", flush=True)
        raise SystemExit(1)
print("ALL-DONE", flush=True)
