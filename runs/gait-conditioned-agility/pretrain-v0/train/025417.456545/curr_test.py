import torch

anchor_pos = torch.tensor([3,3,3], dtype=torch.float)
base_pos = torch.tensor([0,0,1], dtype=torch.float)

ab= anchor_pos - base_pos

norm_ab = torch.norm(ab)

unit_vector_ab = ab / norm_ab

print(unit_vector_ab)