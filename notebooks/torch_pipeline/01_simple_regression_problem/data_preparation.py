
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tranform the data from numpy array to torch tensor
x_train_tensor = torch.as_tensor(x_train).float().to(device)
y_train_tensor = torch.as_tensor(y_train).float().to(device)
