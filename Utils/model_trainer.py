from torch.utils.data import DataLoader
import torch
from datetime import datetime
import os


class ModelTrainer():
    def __init__(self, save_model_path=None):
        self.save_model_path = save_model_path

    def train_model(self, net, train_loader:DataLoader, val_loader:DataLoader,
                    loss_fn, optimizer, device:torch.device, epchos:int, scheduler=None):
        self.net = net
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        best_vloss = 0

        for epoch in range(epchos):
            print(f'EPOCH {epoch + 1}:')

            avg_loss = self.train_one_epoch()

            avg_vloss, vaccuracy = self.validate_model(
                self.net,
                val_loader,
                self.loss_fn,
                self.device
            )

            if scheduler:
                scheduler.step()

            # Track best performance, and save the model's state
            if (avg_vloss > best_vloss) and (self.save_model_path):
                best_vloss = avg_vloss
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_path = f"{self.save_model_path}-{timestamp}.pt"
                torch.save(self.net.state_dict(), model_path)

            print(f'LOSS train {avg_loss} valid {avg_vloss} ACCURACY {vaccuracy}')

    def train_one_epoch(self):
        self.net.train()

        running_loss = 0
        last_loss = 0
        num_batches = len(self.train_loader)

        for i, data in enumerate(self.train_loader):
            inputs, refrence = data[0].to(self.device), data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss_fn(outputs, refrence)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if (i % (num_batches // 5)) == ((num_batches // 5) - 1):
                last_loss = running_loss / (num_batches // 5)
                print(f'\tbatch {i + 1} loss: {last_loss}')
                running_loss = 0

        return last_loss

    def validate_model(self, net, val_loader:DataLoader,
                       loss_fn, device:torch.device):
        net.eval()
        
        running_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, reference = data[0].to(device), data[1].to(device)

                outputs = net(inputs)
                
                loss = loss_fn(outputs, reference)
                running_loss += loss.item()

                predictions = torch.max(outputs, 1)[1].to(device)
                correct += (predictions == reference).sum()
                total += len(reference)

        avg_loss = running_loss / (i + 1)
        accuracy = correct / total

        return avg_loss, accuracy

    def save_checkpoint(epoch, model, optimizer, path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at epoch {epoch}")