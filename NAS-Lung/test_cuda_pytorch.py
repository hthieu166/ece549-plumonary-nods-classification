import torch

print(torch.version.cuda)
print("Total cuda devices", torch.cuda.device_count())