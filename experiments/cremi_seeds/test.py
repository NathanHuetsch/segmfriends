from segmfriends.datasets.cremi_seeds import CremiSeedDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib

path = '/home/nhuetsch/Desktop/Data/cremi2012/sampleA_network_v0.h5'
a = CremiSeedDataset(path)

loader = DataLoader(a, batch_size=1)

for batch in loader:
    input, target = batch
    break
print(input.shape)
print(target.shape)

matplotlib.use('module://backend_interagg')
plt.imshow(input[0,0,0].numpy())
plt.show()

plt.imshow(input[0,1,0].numpy())
plt.show()

plt.imshow(target[0,0,0].numpy())
plt.show()