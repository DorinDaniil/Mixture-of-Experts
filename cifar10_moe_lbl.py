import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from src.moe_lbl import MoELayer, load_balancing_loss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 10
INPUT_SIZE = 32 * 32 * 3
HIDDEN_SIZE = 128
NUM_EXPERTS = 6

class MoEClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, num_classes):
        super().__init__()
        self.moe = MoELayer(input_dim, hidden_dim, num_experts)
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: (batch, input_dim)
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        moe_out, gate_probs, expert_indices = self.moe(x)
        logits = self.head(moe_out.squeeze(1))
        return logits, gate_probs, expert_indices

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    net = MoEClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_EXPERTS, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Тренировка
    for epoch in range(3):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.view(inputs.shape[0], -1).to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs, gate_probs, expert_indices = net(inputs)
            ce_loss = criterion(outputs, labels)
            lbl_loss = load_balancing_loss(gate_probs, expert_indices, NUM_EXPERTS, alpha=0.01)
            loss = ce_loss + lbl_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch+1}, {i+1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print('Finished Training')

    # Оценка
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.view(images.shape[0], -1).to(DEVICE)
            labels = labels.to(DEVICE)
            outputs, _, _ = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.1f}%')

if __name__ == "__main__":
    main()