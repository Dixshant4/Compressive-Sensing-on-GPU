import matplotlib.pyplot as plt
import torch
import torch.cuda as cuda
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms

"""Generating 500 sparse images and their corresponding measurement vector for our data set"""
# S=100
pixels=10 #number of pixels in single dimention
S = 10 #(pixels**2)/40 #approximate sparsity (number of non zero pixels)
p=float(S)/float(pixels**2)
m = 30# number of measurements int(2*(S)*np.log(n/S)) #number of measurements. the formula is the one specefied in the candes paper
epsilon=0.0001

# Defining the sensing matrix
def sensing_matrix(m,n):

    return torch.bernoulli(torch.full((m, n), 0.5))

output_data = []
input_data = []
data_size = 1000
elts_in_input = m
for i in range(data_size):
    sparsity_generator = np.random.binomial(1, p, (pixels,pixels))  # generates a matrix with values 0 or 1 taken from a binomial dist with 1 trial
    x=np.random.uniform(low=0.0, high=1.0, size=(pixels,pixels))  # generates a random matrix x where each element is independently sampled from a uniform distribution between 0.0 and 1.0
    x=sparsity_generator*x
    x=x.flatten()

    output_data.append(x)

    n=x.size #size of vector x i.e total number of pixels
    # print("number of pixels: " +str(n)+ "sparsity is: " + str(S)+  " and number of sampled mesurements: " +str(m))

    """ GPU configuration"""
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    #
    x_tensor = torch.from_numpy(x).view(n,1).to(torch.float32).to(device)
    z = (torch.randn(m).view(m,1) * (epsilon / 3000)).to(device)
    matrix = sensing_matrix(m,n).to(device)

    y = matrix @ x_tensor # A vector where each component is the power sensed by the detector
    input_data.append(y)

output_data = np.array(output_data)
# print(output_data)
input_data = np.reshape(np.array(input_data), (data_size,elts_in_input))
# print(f'input data: {input_data}')

""" Define the training set"""
Y_train, Y_temp, x_train, x_temp = train_test_split(input_data, output_data, test_size=0.4, random_state=42)
Y_val, Y_test, x_val, x_test = train_test_split(Y_temp, x_temp, test_size=0.5, random_state=42)

# Here, Y_train is set of y measurement vector and x_train are the coresponding image vector for each y in Y.
# Together, they form thr training data set. Similarly for Y_val and Y_tes.
# print(f'set of measurement_vectors: {Y_test}, {len(Y_test)}')

# Convert data to PyTorch tensors
Y_train = torch.tensor(Y_train, dtype=torch.float32)
x_train = torch.tensor(x_train, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)


"""Build the neural network architecture: Model 1"""
class CNN1(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN1, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(11,11), padding=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(7,7), padding=3)
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(11,11), padding=5)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1))
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(7,7), padding=3)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 1, 20, 20)  # Reshape to match the feature map size
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        return x.view(-1)


# input_size = m
# output_size = pixels**2
# model1 = CNN1(input_size, output_size)

# This is the convolutional neural network architecture from the ReconNet paper

"""Build the neural network architecture: Model 2. Similar to a decoder in Auto encoders"""
class CNN2(nn.Module):
    def __init__(self, input_size, hiddenlayer_1, hiddenlayer_2, output_size):
        super(CNN2, self).__init__()
        self.fc1 = nn.Linear(input_size, hiddenlayer_1)
        self.fc2 = nn.Linear(hiddenlayer_1, hiddenlayer_2)
        self.fc3 = nn.Linear(hiddenlayer_2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.sigmoid(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        return x


"""Evaluating the model on the validation set"""
def evaluate(model, Y_val, x_val, lamb):
    model.eval()
    validloss = 0
    cri = nn.MSELoss()

    for i in range(len(Y_val)):
        output = model(Y_val[i])
        target = x_val[i]
        loss = torch.norm(Y_val[i]-sensing_matrix(m,pixels**2)@output, p=2) + lamb*torch.norm(output, p=1)
        validloss += loss.item()

    return validloss

input_size = m
output_size = pixels**2
hiddenlayer_1 = 40
hiddenlayer_2 = 80
model2 = CNN2(input_size, hiddenlayer_1, hiddenlayer_2, output_size)

"""Training section of the model """
# Tell PyTorch you are training the model.
model2.train()
# Define optimizers and loss function.
optimizer = optim.Adam(model2.parameters())
num_epochs = 10
lamb = 1
cri = nn.MSELoss()
train_loss_array = []
valid_loss_array = []
for epoch in range(num_epochs): # iterating over the whole data set 10 times
    train_loss = 0

    for i in range(len(Y_train)): # iterating over every data point in the training data set
        optimizer.zero_grad()
        outputs = model2(Y_train[i])
        # print(f'outputs predicted during training stage: {outputs.shape}')
        # loss = torch.sum((outputs - x_train[i])**2)
        loss = torch.norm(Y_train[i]-sensing_matrix(m,pixels**2)@outputs, p=2) + lamb*torch.norm(outputs, p=1) # cri(outputs, x_train[i])
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    train_loss_array.append(train_loss)
    valid_loss = evaluate(model2, Y_val, x_val, lamb)
    valid_loss_array.append(valid_loss)
# Save the trained model
# torch.save(model2.state_dict(), 'custom_cnn_model.pth'


plt.plot(range(num_epochs), np.array(train_loss_array)/(num_epochs), label="train loss")
plt.plot(range(num_epochs), np.array(valid_loss_array)/(num_epochs), label="valid loss")
plt.xlabel("num_epochs")
plt.ylabel("loss")
plt.legend()
plt.show()


"""Seeing the predected vs actual output"""
# Make predictions on the test data

new_input = Y_test[2]
predicted_output = model2(new_input)
print(predicted_output)
actual_vec = x_test[2].detach().numpy()
print(actual_vec)
pred_vec = predicted_output.detach().numpy()
# mse = np.linalg.norm(actual_vec-pred_vec)
# print(f"L2 norm: {mse}")
# 'predicted_output' will contain the predicted output vector for the new input


# construct the image of actual_vec and pred_vec
actual_vec = np.reshape(actual_vec,(pixels,pixels))
pred_vec = np.reshape(pred_vec,(pixels,pixels))

img_recon = Image.fromarray(np.uint8(actual_vec * 255),mode='L')
img_recon.save("original.png")


img_recon = Image.fromarray(np.uint8(pred_vec * 255),mode='L')
img_recon.save("reconstructed_torch1.png")
