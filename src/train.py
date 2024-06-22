import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import classification_report

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.data.road import Road
from src.data.transform_road import TransformRoad
from src.model.resnet import Resnet

learning_rate = 3e-4    # Karpathy constance
batch_size = 16
num_workers = 2
epochs = 5
device = 'cpu'


def get_dataset():
    train_set = Road(data_type='train')
    val_set = Road(data_type='val')
    test_set = Road(data_type='test')

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    train_set = TransformRoad(train_set, transform)
    val_set = TransformRoad(val_set, transform)
    test_set = TransformRoad(test_set)

    return train_set, val_set, test_set


def get_model(model_type='resnet18', pretrain: bool = True):
    model_name = ''.join([i for i in model_type if not i.isdigit()])    # remove numbers
    model_num = model_type[len(model_name):]

    # if model_name == 'resnet':
    # elif model_name == 'alexnet':
    return Resnet(model_num, pretrain)
        

def train_loop(dataloader, model, loss_fn, optimizer, device='cpu'):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def val_loop(dataloader, model, loss_fn, device='cpu'):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            pred = F.softmax(pred, dim=1)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def test_loop(dataloader, model, loss_fn, device='cpu'):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss = 0
    ground_truth = []
    predictions = []
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss += loss_fn(pred, y).item()

            pred = F.softmax(pred, dim=1)
            pred = pred.argmax(1)

            ground_truth += y.cpu().tolist()
            predictions += pred.cpu().tolist()

    loss /= num_batches
    print(f"Avg loss: {loss:>8f} \n")
    print(classification_report(ground_truth, predictions))


def main(epochs=50):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() and device =='cuda' else "cpu")
    
    train_set, val_set, test_set = get_dataset()
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = get_model().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)    

    print("Training...")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, DEVICE)
        val_loop(val_dataloader, model, loss_fn, DEVICE)

    print("\nTesting...")
    test_loop(test_dataloader, model, loss_fn, DEVICE)


if __name__ == "__main__":
    main()