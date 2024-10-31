from torch.utils.data import DataLoader
import torch

class ModelTrainer():
    def __init__(self):
        pass

    def train_model(self, net, train_loader:DataLoader, val_loader:DataLoader,
                    loss_fn, optimizer, device:torch.device, epchos:int):
        self.net = net
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        for epoch in range(epchos):            
            self.train_one_epoch()
            
            self.validate_model(
                self.net,
                val_loader,
                self.loss_fn,
                self.device
            )

    def train_one_epoch(self):
        self.net.train(True)

        for i, data in enumerate(self.train_loader):
            inputs, refrence = data[0].to(self.device), data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss_fn(outputs, refrence)
            loss.backward()
            self.optimizer.step()

    def validate_model(self, net, val_loader:DataLoader,
                       loss_fn, device:torch.device):
        net.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, refrence = data[0].to(device), data[1].to(device)

                outputs = net(inputs)
                loss = loss_fn(outputs, refrence)
                
                running_loss += loss
        avg_loss = running_loss / (i + 1)
        return avg_loss