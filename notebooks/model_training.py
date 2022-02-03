# Define number of epochs 
n_epochs=1000

for epoch in range(n_epochs): 
    model.train()  #2)
    # Step 1: Computes the model's predicted output - forward pass
    # No more manula prediction 
    #yhat = b + w*x_train_tensor 
    yhat = model(x_train_tensor)  #3)
    
    # Step 2: Computes the loss
    # No more manual loss
    # error = (yhat - y_train_tensor)
    # loss = (error**2).mean()
    loss = loss_fn(yhat, y_train_tensor) #2
    
    # Step 3: Computes gradients for both 'b' and 'w' parameters
    loss.backward()
    
    # Step 4: Updates parameters using gradients and the learning rate
    # No more manual update
    # with torch.no_grade():
    #    b-=lr*b.grad
    #    w-=lr*w.grad
    optimizer.step()
    
    
    # Graident Zeroing
    # No more telling pytorch to let gradients go 
    #b.grad.zero()
    #w.grad.zero()
    optimizer.zero_grad()

