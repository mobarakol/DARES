from transformers import AutoImageProcessor, AutoModelForDepthEstimation, DepthAnythingForDepthEstimation
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import math
from torch.nn.parameter import Parameter

class _LoRA_qkv(nn.Module):
    """In Dinov2 it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            w: nn.Module,
            linear_a: nn.Module,
            linear_b: nn.Module
    ):
        super().__init__()
        self.w = w
        self.linear_a = linear_a
        self.linear_b = linear_b
        self.dim = w.in_features

    def forward(self, x):
        W = self.w(x)
        residual = W.clone()
        deltaW = self.linear_b(self.linear_a(x))

        W += deltaW
        return W

class DepthAnythingDepthEstimationHead(nn.Module):

    def __init__(self, model_head):
        super().__init__()
        self.conv1 = model_head.conv1
        self.conv2 = model_head.conv2
        self.activation1 = nn.ReLU()
        self.conv3 = model_head.conv3
        self.activation2 = nn.Sigmoid()

    def forward(self, hidden_states, height, width):
        predicted_depth = self.conv1(hidden_states)
        predicted_depth = nn.functional.interpolate(
            predicted_depth,
            (int(height), int(width)),
            mode="bilinear",
            align_corners=True,
        )
        predicted_depth = self.conv2(predicted_depth)
        predicted_depth = self.activation1(predicted_depth)
        predicted_depth = self.conv3(predicted_depth)
        predicted_depth = self.activation2(predicted_depth)
        return predicted_depth

class LoRAInitializer:
    def __init__(self, model, r=[14,14,12,12,10,10,8,8,8,8,8,8], lora=['q', 'v']):
        self.model = model
        self.r = r
        self.lora = lora
        self.w_As = []
        self.w_Bs = []
        self.initialize_lora()

    def initialize_lora(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(self.model.backbone.encoder.layer):
            dim = blk.attention.attention.query.in_features

            if 'q' in self.lora:
                w_q = blk.attention.attention.query
                w_a_linear_q = nn.Linear(dim, self.r[t_layer_i], bias=False)
                w_b_linear_q = nn.Linear(self.r[t_layer_i], dim, bias=False)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                blk.attention.attention.query = _LoRA_qkv(w_q, w_a_linear_q, w_b_linear_q)

            if 'v' in self.lora:
                w_v = blk.attention.attention.value
                w_a_linear_v = nn.Linear(dim, self.r[t_layer_i], bias=False)
                w_b_linear_v = nn.Linear(self.r[t_layer_i], dim, bias=False)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                blk.attention.attention.value = _LoRA_qkv(w_v, w_a_linear_v, w_b_linear_v)

            if 'k' in self.lora:
                w_k = blk.attention.attention.key
                w_a_linear_k = nn.Linear(dim, self.r[t_layer_i], bias=False)
                w_b_linear_k = nn.Linear(self.r[t_layer_i], dim, bias=False)
                self.w_As.append(w_a_linear_k)
                self.w_Bs.append(w_b_linear_k)
                blk.attention.attention.key = _LoRA_qkv(w_k, w_a_linear_k, w_b_linear_k)

        self.reset_parameters()
        print("LoRA params initialized!")

    def reset_parameters(self):
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


class Dares(nn.Module):
    def __init__(self, r = [14,14,12,12,10,10,8,8,8,8,8,8], lora = ['q', 'v']):
        super(Customised_DAM, self).__init__()
        model = DepthAnythingForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.r = r
        self.lora = lora
        self.config = model.config
        self.backbone = model.backbone

        # Initialize LoRA parameters
        self.lora_initializer = LoRAInitializer(model, r, lora)

        self.neck = model.neck
        model_head = model.head
        self.head = DepthAnythingDepthEstimationHead(model_head)
        model.post_init()

    def save_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        decode_head_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.decode_head, torch.nn.DataParallel) or isinstance(self.decode_head, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.decode_head.module.state_dict()
        else:
            state_dict = self.decode_head.state_dict()
        for key, value in state_dict.items():
            decode_head_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **decode_head_tensors}
        torch.save(merged_dict, filename)

        print('saved lora parameters to %s.' % filename)

    def load_parameters(self, filename: str, device: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename, map_location=device)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        decode_head_dict = self.decode_head.state_dict()
        decode_head_keys = decode_head_dict.keys()

        # load decode head
        decode_head_keys = [k for k in decode_head_keys]
        decode_head_values = [state_dict[k] for k in decode_head_keys]
        decode_head_new_state_dict = {k: v for k, v in zip(decode_head_keys, decode_head_values)}
        decode_head_dict.update(decode_head_new_state_dict)

        self.decode_head.load_state_dict(decode_head_dict)

        print('loaded lora parameters from %s.' % filename)

    def forward(self, pixel_values):
        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=None, output_attentions=None
        )
        hidden_states = outputs.feature_maps
        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size
        hidden_states = self.neck(hidden_states, patch_height, patch_width)
        outputs = {}
        outputs[("disp", 0)] = self.head(hidden_states[3], height, width)
        outputs[("disp", 1)] = self.head(hidden_states[2], height/2, width/2)
        outputs[("disp", 2)] = self.head(hidden_states[1], height/4, width/4)
        outputs[("disp", 3)] = self.head(hidden_states[0], height/8, width/8)
        return outputs
