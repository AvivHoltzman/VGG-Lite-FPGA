import torch, numpy as np
from vgg_lite_model import VGG_Lite_INT8

model = VGG_Lite_INT8()
model.load_state_dict(torch.load('vgg_lite_int8_best.pth', map_location='cpu'))
model.eval()

def wb(layer):
    W = layer.quant_weight().value.detach().numpy().astype(np.float32)
    b = (layer.bias.detach().numpy().astype(np.float32)
         if layer.bias is not None else np.zeros(W.shape[0], np.float32))
    return W, b

W1, b1 = wb(model.block1[0])      # conv1  3 -> 16
W2, b2 = wb(model.block1[2])      # conv2 16 -> 16
W3, b3 = wb(model.block2[0])      # conv3 16 -> 32
W4, b4 = wb(model.block2[2])      # conv4 32 -> 32
Wf, bf = wb(model.classifier[2])  # fc   128 -> 10

np.savez_compressed('sw_weights.npz',
                    W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3,
                    W4=W4, b4=b4, Wf=Wf, bf=bf)

for n, W in [("conv1",W1),("conv2",W2),("conv3",W3),("conv4",W4),("fc",Wf)]:
    print(f"  {n:6s} {W.shape}")
print("saved sw_weights.npz")
