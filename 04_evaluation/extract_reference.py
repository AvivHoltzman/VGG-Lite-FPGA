import pickle, os, sys, numpy as np
from qonnx.core.modelwrapper import ModelWrapper
try:    from qonnx.core.onnx_exec import execute_onnx
except: from finn.core.onnx_exec import execute_onnx

CIFAR = "/home/avivh/finn/cifar_data"          # extracted cifar-10-batches-py folder
BUILD = "/home/avivh/finn/deploy/build_vgg_lite_int8_bitfile_final9283rce6/intermediate_models"
N     = int(sys.argv[1]) if len(sys.argv) > 1 else 200

MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32)
STD  = np.array([0.2023, 0.1994, 0.2010], np.float32)
ACC_IN, ACC_OUT, CUT = "MultiThreshold_0_out0", "MaxPoolNHWC_1_out0", "MaxPoolNHWC_1"

with open(os.path.join(CIFAR, "test_batch"), "rb") as f:
    d = pickle.load(f, encoding="bytes")
X = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
Y = np.array(d[b"labels"], np.int64)

m    = ModelWrapper(os.path.join(BUILD, "step_streamline.onnx"))
gin  = m.graph.input[0].name
gout = m.graph.output[0].name

# ---------- run N images through the streamlined graph ----------
imgs, a_in, a_out, logits = [], [], [], []
for i in range(N):
    x = ((X[i].astype(np.float32)/255.0 - MEAN)/STD).transpose(2, 0, 1)[None]
    ctx = execute_onnx(m, {gin: x}, return_full_exec_context=True)
    imgs.append(X[i]); a_in.append(ctx[ACC_IN][0])
    a_out.append(ctx[ACC_OUT][0]); logits.append(ctx[gout][0])
    if (i+1) % 25 == 0: print(f"  {i+1}/{N}", flush=True)

imgs   = np.stack(imgs).astype(np.uint8)
a_in   = np.stack(a_in).astype(np.uint8)      # (N,32,32,16)  -> accelerator input
a_out  = np.stack(a_out).astype(np.uint8)     # (N,8,8,32)    -> expected output
logits = np.stack(logits).astype(np.float32)
labels = Y[:N]

# ---------- extract the software tail (everything after the partition) ----------
cut  = [i for i, n in enumerate(m.graph.node) if n.name == CUT][0]
perm, muls, fcW, bias, kern = [0, 3, 1, 2], [], None, None, 4
for n in m.graph.node[cut+1:]:
    ini = [np.asarray(m.get_initializer(i)) for i in n.input
           if m.get_initializer(i) is not None]
    if   n.op_type == "Transpose":   perm = list(n.attribute[0].ints)
    elif n.op_type == "Mul":         muls += ini
    elif n.op_type == "MatMul":      fcW  = ini[0] if ini else None
    elif n.op_type == "Add":         bias = ini[0] if ini else None
    elif n.op_type == "AveragePool":
        for a in n.attribute:
            if a.name == "kernel_shape": kern = int(a.ints[0])
mul0 = muls[0] if len(muls) > 0 else np.float32(1.0)
mul1 = muls[1] if len(muls) > 1 else np.float32(1.0)

# ---------- self check: rebuild the tail in pure numpy ----------
def tail(feat_nhwc):
    f = feat_nhwc.astype(np.float32).transpose(perm) * mul0
    n, c, h, w = f.shape
    f = f.reshape(n, c, h//kern, kern, w//kern, kern).mean(axis=(3, 5))
    return (f.reshape(n, -1) @ fcW) * mul1 + bias

mine = tail(a_out)
err  = np.abs(mine - logits).max()
print(f"\nSELF-CHECK  max logit error = {err:.6f}   "
      f"pred match = {100*(mine.argmax(1)==logits.argmax(1)).mean():.2f}%")

np.savez_compressed(os.path.join(BUILD, "reference_data.npz"),
    imgs=imgs, acc_in=a_in, acc_out=a_out, ref_logits=logits, labels=labels,
    perm=np.array(perm), mul0=mul0, mul1=mul1, fcW=fcW, bias=bias,
    kern=np.array(kern))

print(f"\nSaved reference_data.npz  ({N} images)")
print(f"  accelerator input : {a_in.shape}  range {a_in.min()}..{a_in.max()}")
print(f"  expected output   : {a_out.shape} range {a_out.min()}..{a_out.max()}")
print(f"  REFERENCE ACCURACY: {100*(logits.argmax(1)==labels).mean():.2f}%")
print(f"  tail params: perm={perm} kern={kern} mul0{np.shape(mul0)} "
      f"fcW{np.shape(fcW)} bias{np.shape(bias)}")
