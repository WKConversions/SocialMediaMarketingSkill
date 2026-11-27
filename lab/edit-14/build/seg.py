"""Salient-object masks with BiRefNet-lite (MIT, onnx-community/BiRefNet_lite-ONNX): seg(img_bgr) -> float mask 0..1."""
import numpy as np, cv2, onnxruntime as ort, os
_S = None
def _sess():
    global _S
    if _S is None:
        o = ort.SessionOptions(); o.intra_op_num_threads = 4; o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _S = ort.InferenceSession(os.path.join(os.path.dirname(__file__), "..", "models", "birefnet_lite.onnx"), o, providers=["CPUExecutionProvider"])
    return _S
def seg(img, lift=True):
    h, w = img.shape[:2]; x = img
    if lift:                                   # the footage is very dark: lift it for the model only
        g = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); l = g[..., 0].astype(np.float32); p = np.percentile(l, 99.5) + 1
        g[..., 0] = np.clip(255 * (l / p) ** 0.55, 0, 255).astype(np.uint8); x = cv2.cvtColor(g, cv2.COLOR_LAB2BGR)
    r = cv2.resize(x, (1024, 1024), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
    r = (r - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    s = _sess(); out = s.run(None, {s.get_inputs()[0].name: r.transpose(2, 0, 1)[None]})[0]
    m = 1 / (1 + np.exp(-out[0, 0])) if out.min() < 0 or out.max() > 1 else out[0, 0]
    return cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
