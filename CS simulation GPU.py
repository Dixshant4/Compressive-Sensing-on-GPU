import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import numpy as np
from PIL import Image
import cvxpy as cp
import time
import torch.optim as optim
import matplotlib.pyplot as plt



# S=100
pixels=50 #number of pixels in single dimention
S = (pixels**2)/40 #approximate sparsity (number of non zero pixels)
p=float(S)/float(pixels**2)
sparsity_generator = np.random.binomial(1, p, (pixels,pixels))  # generates a matrix with values 0 or 1 taken from a binomial dist with 1 trial
x=np.random.uniform(low=0.0, high=1.0, size=(pixels,pixels))  # generates a random matrix x where each element is independently sampled from a uniform distribution between 0.0 and 1.0
x=sparsity_generator*x
print(x)
print(sparsity_generator)


#save the generated image

img = Image.fromarray(np.uint8(x * 255),mode='L')
img.save("orignal.png")
#flatten the image from 2d to 1d array

x=x.flatten()
print("original x")
print(x)

n=x.size #size of vector x i.e total number of pixels (100)

m= int(2*(S)*np.log(n/S)) #number of measurements. the formula is the one specefied in the candes paper
print("number of pixels: " +str(n)+ "sparsity is: " + str(S)+  " and number of sampled mesurements: " +str(m))

epsilon=0.0001

def sensing_matrix(m,n):
    return torch.bernoulli(torch.full((m, n), 0.5))

sensing_matrix(m,n).size()



B = sensing_matrix(m,n)
w = B.numpy()@x


num_cons=1
p=1
n_i=m

""" GPU configuration"""
device = torch.device('cuda' if cuda.is_available() else 'cpu')
#
#
x_tensor = torch.from_numpy(x).view(n,1).to(torch.float32).to(device)
z = (torch.randn(m).view(m,1) * (epsilon / 3000)).to(device)
matrix = sensing_matrix(m,n).to(device)
print(x_tensor)
print(x_tensor.size())
print(torch.count_nonzero(x_tensor))
print(matrix)
print(z)

# Start the timer
start_time = time.time()
y = matrix @ x_tensor # + z
print(y.size())
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")

print(y)



A = matrix.to(device)
y = y.to(device)

# Start the timer
start_time = time.time()

# Define the optimization problem
x = torch.nn.Parameter(torch.zeros(n, 1, requires_grad=True).to(device))  # Variable to optimize


# Define the optimizer
optimizer = optim.Adam([x], lr=0.000007, betas=(0.95, 0.9995))
losses = []
epoch = []
l1_losses = []
lamb = 1
# Optimization loop
for i in range(1000):
    optimizer.zero_grad()
    loss = torch.norm(A@x-y, p=2) + lamb*torch.norm(x, p=1)
    loss.backward()
    optimizer.step()
    losses.append(loss.cpu().detach().numpy())
    l1_losses.append(torch.norm(x - x_tensor, p=1))
    epoch.append(i)


# Get the optimized solution
x_solution = x.detach().cpu().numpy()
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")
plt.plot(epoch, losses)
plt.show()
l1_np = [tensor.detach().cpu().numpy() for tensor in l1_losses]
plt.plot(epoch, l1_np)
plt.show()
print(torch.count_nonzero(x))

print(torch.abs(x_tensor).sum())
print(np.abs(x_solution).sum())
print("Optimized x:")
print(x_solution)
print("original x:")
print(x_tensor)
y = A@x_tensor
y_recon = A@x

print(y-y_recon)

# print()

# Move variables to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x_solution = x_solution - np.min(x_solution)
x_solution = x_solution/np.max(x_solution)
loss = np.linalg.norm(x_solution - x_tensor.cpu().numpy())
print(f"loss: {loss}")




x = np.reshape(x_solution,(pixels,pixels))
img_recon = Image.fromarray(np.uint8(x * 255),mode='L')
img_recon.save("reconstructed_torch.png")

