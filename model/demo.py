import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ETR_NLP import avg_pool

# Create a sample image tensor
image = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                     [5.0, 6.0, 7.0, 8.0],
                     [9.0, 10.0, 11.0, 12.0],
                     [13.0, 14.0, 15.0, 16.0]], dtype=torch.float32)


pool = avg_pool(3)
result = pool(image)
print(result)