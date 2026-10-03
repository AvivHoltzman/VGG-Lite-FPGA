import torch
import torch.nn as nn
import brevitas.nn as qnn

class VGG_Lite_INT8(nn.Module):
    def __init__(self):
        super(VGG_Lite_INT8, self).__init__()
        
        # Define bits-width for quantization - 8 bits
        self.bit_width = 8
        
        # Block 1: 2 Conv (3x3, 16 Kernels) + ReLU + Maxpool (16x16)
        self.block1 = nn.Sequential(
            qnn.QuantConv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, weight_bit_width=self.bit_width),
            qnn.QuantReLU(bit_width=self.bit_width),
            qnn.QuantConv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, weight_bit_width=self.bit_width),
            qnn.QuantReLU(bit_width=self.bit_width),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: 2 Conv (3x3, 32 Kernels) + ReLU + Maxpool (8x8)
        self.block2 = nn.Sequential(
            qnn.QuantConv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, weight_bit_width=self.bit_width),
            qnn.QuantReLU(bit_width=self.bit_width),
            qnn.QuantConv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, weight_bit_width=self.bit_width),
            qnn.QuantReLU(bit_width=self.bit_width),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classification Layer: Partial Pooling + Fully Connected
        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4), # Shrinks 8x8 map to 2x2
            nn.Flatten(),                          # Flatten the vector to 128 features (32 * 2 * 2)
            qnn.QuantLinear(in_features=128, out_features=10, bias=True, weight_bit_width=self.bit_width)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

# Instantiate the model and print its structure
if __name__ == '__main__':
    model = VGG_Lite_INT8()
    print("VGG-Lite INT8 Model initialized successfully!")
    print(model)