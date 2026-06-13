import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64,   shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=1000, shuffle=False)


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTClassifier().to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_epoch(model, train_loader, loss_function, optimizer):
    model.train()
    rloss = 0.0
    correct = 0
    total = 0

    for batch, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        rloss += loss.item()

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch % 100 == 0 and batch > 0:
            avgloss = rloss / 100
            accuracy = 100. * correct / total
            progress = 100. * batch * train_loader.batch_size / len(train_loader.dataset)
            print(f"[{progress:5.1f}%]  loss: {avgloss:.3f}  acc: {accuracy:.1f}%")
            rloss = 0.0


# run a few epochs
for epoch in range(3):
    print(f"\nEpoch {epoch+1}")
    train_epoch(model, train_loader, loss_function, optimizer)