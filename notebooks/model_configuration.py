class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        #Make `b` and `w` be the parameters of the model
        #Wrap them with `nn.Parameter` 
        #self.b = nn.Parameter(torch.randn(1,
        #                                requires_grad=True,
        #                                dtype=torch.float))
        #self.w = nn.Parameter(torch.randn(1,
        #                                requires_grad=True,
        #                                dtype=torch.float))
    def forward(self, x):
        # Compute the outputs /predictions
        return self.linear(x)

# Sets learning rate
lr = 0.1 

# Step 0 : Initialize parameters 'b' and 'w' randomly
torch.manual_seed(42)

# Create a model and send it at once to the device
model = ManualLinearRegression().to(device) # 1)

# Define a SGD optimizer to update the parameters
#optimizer = torch.optim.SGD([b,w], lr=lr)
optimizer = torch.optim.SGD(model.parameters(),lr=lr)


#Define a MSE loss function 
loss_fn = nn.MSELoss(reduction="mean")  
