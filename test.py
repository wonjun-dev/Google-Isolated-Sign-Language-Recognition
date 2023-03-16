import torch
from model import *
from collections import OrderedDict


if __name__ == "__main__":
    ckpt = "/sources/ckpts/test-512/2023-03-14T16-02-15/1/best.pth.tar"

    model = ISLRModel(embed_dim=512, n_head=4, ff_dim=1024)
    trained_state_dict = torch.load(ckpt)["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        name = k
        if "module" in name:
            name = name.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    print(model)
