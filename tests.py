import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers.models.resnet.modeling_resnet import ResNetBasicLayer, ResNetBottleNeckLayer

class ResNetForKoopmanEstimation(torch.nn.Module):
    def __init__(self, model_name="microsoft/resnet-18", out_channels:int=512, out_size: tuple= (1, 1)):
        super().__init__()
        # Load pretrained model
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        # For capturing residual states
        self.pre_residuals = []
        self.post_residuals = []
        self.residuals = []

        self.out_channels = out_channels
        self.out_size = out_size
        # Register hooks on all residual blocks
        for module in self.model.resnet.encoder.modules():
            if isinstance(module, (ResNetBasicLayer, ResNetBottleNeckLayer)):
                module.register_forward_hook(self._hook_fn)

        option_layers = {}
        for channels in [64, 128, 256, 512, 1024, 2048]:
            option_layers[str(channels)] = nn.Sequential(nn.Conv2d(channels, out_channels, kernel_size=1), nn.AdaptiveAvgPool2d(out_size), nn.Flatten(1))

        self.upconvs = nn.ModuleDict(option_layers)

        self.feature_projector = nn.Linear(out_size[0] * out_size[1], 1)


    def _hook_fn(self, module, input, output):
        if len(self.residuals) == 0:
            self.residuals.append(input[0].detach())
            self.residuals.append(output.detach())
        else:
            self.residuals.append(output.detach())

        self.pre_residuals.append(input[0].detach().cpu())
        self.post_residuals.append(output.detach().cpu())

    def forward(self, pixel_values, labels=None):
        # Clear previous states
        self.pre_residuals.clear()
        self.post_residuals.clear()
        output = self.model(pixel_values=pixel_values, labels=labels)
        residual_states = []
        for residual in self.residuals:
            key_channels = residual.shape[1]
            upconv = self.upconvs[str(key_channels)](residual).relu()
            out = self.feature_projector(upconv.view(pixel_values.shape[0], self.out_channels, -1)).relu()
            out = out.flatten(1)
            residual_states.append(out.unsqueeze(0))

        output.hidden_states = torch.cat(residual_states, dim=0)


        return output


if __name__ == "__main__":
    # Load processor and model
    model_name = "microsoft/resnet-34"
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = ResNetForKoopmanEstimation(model_name)

    # Fake CIFAR-100-like input (3x32x32), resized to 224x224 (ImageNet input size)
    images = torch.randn(4, 3, 224, 224)  # Use 224x224 for pretrained ResNet

    # Forward pass
    import pdb; pdb.set_trace()
    outputs = model(images)
    print("Logits:", outputs.logits.shape)
    print("Pre-residual shapes:", [t.shape for t in model.pre_residuals])
    print("Post-residual shapes:", [t.shape for t in model.post_residuals])
    print("Len Pre-residuals-", len(model.pre_residuals))
    print("Len Post-residuals-", len(model.post_residuals))
    print("len residuals", len(model.residuals))
    print("len residuals states", (outputs.hidden_states).shape)
