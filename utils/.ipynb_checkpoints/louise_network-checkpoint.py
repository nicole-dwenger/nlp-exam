class NeuralNet(nn.Module):
    def __init__(self, layers=[10,30,1], cost=nn.BCELoss()):
        super(NeuralNet, self).__init__()
        l = []
        for i in range(len(layers)-1):
            l.append(nn.Linear(layers[i], layers[i+1]))
        self.linear = nn.Sequential(*l)
        self.cost = cost

    def forward(self, x):
        for layer in self.linear:
            x = torch.sigmoid(layer(x))
        return x 

    def fit(self, X, y, epochs=1000, batch_size=100):
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        y = y.reshape(y.size()[0],1)
        optimizer = torch.optim.AdamW(self.parameters())

        for epoch in range(epochs):
            ds = TensorDataset(X, y)
            loader = dataloader.DataLoader(ds, batch_size=batch_size, shuffle=True)

            for batch in loader: #Run through all batches
                X, y = batch
                y_hat = self.forward(X)

                # backward
                loss = self.cost(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # print
            if (epoch + 1) % 100 == 0:
                print(f"epoch: {epoch+1}, loss = {loss.item():.4f}")

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float)
        forw = self.forward(X)
        predictions = [1 if p > 0.5 else 0 for p in forw]
        return predictions
