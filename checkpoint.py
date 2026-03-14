import os
import glob
import torch
from datetime import datetime

class CheckpointManager:
    def __init__(self, save_dir, max_checkpoints=5):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch, metrics):
        """ Saves a checkpoint of the model, optimizer, epoch, and metrics. """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics
        }
        today = datetime.now().strftime("%Y-%m-%d")
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}_{today}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """ Removes old checkpoints if the number exceeds max_checkpoints. """
        checkpoints = sorted(glob.glob(os.path.join(self.save_dir, 'checkpoint_epoch_*.pth')), key=os.path.getmtime)
        while len(checkpoints) > self.max_checkpoints:
            os.remove(checkpoints[0])
            print(f"Removed old checkpoint: {checkpoints[0]}")
            checkpoints.pop(0)
        
    def load_checkpoint(self, model, optimizer, checkpoint_path):
        """ Loads a checkpoint and restores the model, optimizer, epoch, and metrics. """
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) if optimizer is not None else None
        epoch = checkpoint['epoch']
        metrics = checkpoint['metrics']
        print(f"Checkpoint loaded: {checkpoint_path} (Epoch {epoch})")
        return epoch, metrics

# checkpoint_manager = CheckpointManager(save_dir='/root/Project/brick_vidgen/vnclip/checkpoints', max_checkpoints=10)