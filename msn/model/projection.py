import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


"""
encoder.fc = None
fc = OrderedDict([])
fc["fc1"] = torch.nn.Linear(emb_dim, hidden_dim)
if use_bn:
    fc["bn1"] = torch.nn.BatchNorm1d(hidden_dim)
fc["gelu1"] = torch.nn.GELU()
fc["fc2"] = torch.nn.Linear(hidden_dim, hidden_dim)
if use_bn:
    fc["bn2"] = torch.nn.BatchNorm1d(hidden_dim)
fc["gelu2"] = torch.nn.GELU()
fc["fc3"] = torch.nn.Linear(hidden_dim, output_dim)
encoder.fc = torch.nn.Sequential(fc)

for m in encoder.modules():
    if isinstance(m, torch.nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)

"""


def get_projection():
    inputs = layers.Input(shape=())