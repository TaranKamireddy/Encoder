import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main(inputFile='data.csv'):
    df = pd.read_csv(inputFile, header=None)
    X = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 1].values.reshape(-1, 1)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(1, 32), 
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1) 
            )

        def forward(self, x):
            return self.model(x)

    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 300
    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predictions = scaler_y.inverse_transform(predictions.numpy())
        actual = scaler_y.inverse_transform(y_test.numpy())

    plt.scatter(scaler_x.inverse_transform(X_test.numpy()), actual, label='Actual')
    plt.scatter(scaler_x.inverse_transform(X_test.numpy()), predictions, label='Predicted', alpha=0.7)
    plt.xlabel('Encoder Input')
    plt.ylabel('Wrist Angle')
    plt.title('Predicted vs Actual')
    plt.legend()
    plt.show()

    def predict_y(x_input):
        if isinstance(x_input, (int, float)):
            x_input = [x_input]

        x_scaled = scaler_x.transform(np.array(x_input).reshape(-1, 1))
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            y_pred = model(x_tensor)
        
        y_unscaled = scaler_y.inverse_transform(y_pred.numpy())
        
        return y_unscaled.flatten() if len(y_unscaled) > 1 else y_unscaled[0][0]


if __name__ == "__main__":
    main()