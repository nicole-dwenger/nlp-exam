## Script for Linear Regression Class

class Linear_Regression(torch.nn.Module):
    def __init__(self, inputSize, outputSize): # 300,1
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
        
    def forward(self, x):
        out = self.linear(x)
        return out    
    
    def fit(self, X_train: np.array, y_train: np.array):
        
        epochs = 10
        learning_rate = 0.01
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch += 1
            
            # Convert numpy array to torch Variable
            inputs = torch.from_numpy(X).requires_grad_()
            labels = torch.from_numpy(y)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad() 

            # Forward to get output
            outputs = model(inputs)

            # Calculate Loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            print('epoch {}, loss {}'.format(epoch, loss.item()))

    
    