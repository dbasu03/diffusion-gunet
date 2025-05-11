import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.nn.functional import mse_loss
from math import log10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Haze4KDataset(Dataset):
    def __init__(self, hazy_dir, gt_dir, transform=None):
        self.hazy_dir = hazy_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.hazy_images = sorted(os.listdir(hazy_dir))
        self.gt_images = sorted(os.listdir(gt_dir))
    
    def __len__(self):
        return len(self.hazy_images)
    
    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])
        
        hazy_img = Image.open(hazy_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        
        if self.transform:
            hazy_img = self.transform(hazy_img)
            gt_img = self.transform(gt_img)
        
        return hazy_img, gt_img

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_hazy_dir = "/kaggle/input/haze4k-t/Haze4K-T/IN"
train_gt_dir = "/kaggle/input/haze4k-t/Haze4K-T/GT"
val_hazy_dir = "/kaggle/input/haze4k-v/Haze4K-V/IN"
val_gt_dir = "/kaggle/input/haze4k-v/Haze4K-V/GT"

train_dataset = Haze4KDataset(train_hazy_dir, train_gt_dir, transform=transform)
val_dataset = Haze4KDataset(val_hazy_dir, val_gt_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.gate = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # Skip connection with 1x1 conv if dimensions change
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        gate_value = self.sigmoid(self.gate(out))
        out = out * gate_value
        
        out += residual
        out = self.relu(out)
        
        return out

class SelectiveKernelFusion(nn.Module):
    def __init__(self, channels):
        super(SelectiveKernelFusion, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // 8, channels * 2, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x1, x2):
        # Fuse for attention
        combined = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1)], dim=1)
        batch, branches, channels, height, width = combined.size()
        
        fused = x1 + x2
        fused = self.gap(fused)
        fused = self.fc1(fused)
        fused = self.relu(fused)
        fused = self.fc2(fused)
        
        attn_weights = fused.view(batch, 2, channels, 1, 1)
        attn_weights = self.softmax(attn_weights)
        
        out = (combined * attn_weights).sum(dim=1)
        return out

class gUNet(nn.Module):
    def __init__(self, in_channels=7, out_channels=3):  # in_channels=7 (noisy_gt + hazy + t_emb)
        super(gUNet, self).__init__()
        
        self.enc1 = GatedResidualBlock(in_channels, 64)
        self.enc2 = GatedResidualBlock(64, 128)
        self.enc3 = GatedResidualBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = GatedResidualBlock(256, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.sk_fusion3 = SelectiveKernelFusion(256)
        self.dec3 = GatedResidualBlock(256, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.sk_fusion2 = SelectiveKernelFusion(128)
        self.dec2 = GatedResidualBlock(128, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.sk_fusion1 = SelectiveKernelFusion(64)
        self.dec1 = GatedResidualBlock(64, 64)
        
        self.final = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x, t):
        t_emb = t.view(-1, 1, 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
        x = torch.cat([x, t_emb], dim=1)  # Concatenate time embedding
        
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        
        bottleneck = self.bottleneck(self.pool(enc3))
        
        dec3 = self.upconv3(bottleneck)
        dec3 = self.sk_fusion3(dec3, enc3)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = self.sk_fusion2(dec2, enc2)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = self.sk_fusion1(dec1, enc1)
        dec1 = self.dec1(dec1)
        
        return self.final(dec1)

class DiffusionModel:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.beta = torch.linspace(beta_start, beta_end, T).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def forward_process(self, x0, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]
        noise = torch.randn_like(x0).to(device)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise, noise
    
    def sample(self, model, hazy_img, n_steps=1000):
        x = torch.randn_like(hazy_img).to(device)
        model.eval()
        with torch.no_grad():
            for t in reversed(range(n_steps)):
                t_tensor = torch.full((hazy_img.shape[0],), t, device=device, dtype=torch.long)
                predicted_noise = model(torch.cat([x, hazy_img], dim=1), t_tensor / n_steps)
                alpha = self.alpha[t]
                alpha_bar = self.alpha_bar[t]
                beta = self.beta[t]
                
                x = (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
                if t > 0:
                    x += torch.sqrt(beta) * torch.randn_like(x)
        return x

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 1]
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

os.makedirs("model_checkpoints", exist_ok=True)

def train(model, diffusion, train_loader, val_loader, epochs=50, save_interval=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    val_psnrs = []
    best_psnr = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for hazy, gt in train_loader:
            hazy, gt = hazy.to(device), gt.to(device)
            batch_size = hazy.shape[0]
            
            t = torch.randint(0, diffusion.T, (batch_size,), device=device).long()
            
            noisy_gt, noise = diffusion.forward_process(gt, t)
            
            input_img = torch.cat([noisy_gt, hazy], dim=1)  # Condition on hazy image
            predicted_noise = model(input_img, t / diffusion.T)
            
            loss = criterion(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        total_psnr = 0
        
        with torch.no_grad():
            for i, (hazy, gt) in enumerate(val_loader):
                hazy, gt = hazy.to(device), gt.to(device)
                batch_size = hazy.shape[0]
                
                t = torch.randint(0, diffusion.T, (batch_size,), device=device).long()
                noisy_gt, noise = diffusion.forward_process(gt, t)
                input_img = torch.cat([noisy_gt, hazy], dim=1)
                predicted_noise = model(input_img, t / diffusion.T)
                
                loss = criterion(predicted_noise, noise)
                val_loss += loss.item()
                
                dehazed = diffusion.sample(model, hazy, n_steps=100)  # Use fewer steps for faster validation
                
                dehazed_norm = (dehazed + 1) / 2  # Convert from [-1,1] to [0,1]
                gt_norm = (gt + 1) / 2          # Convert from [-1,1] to [0,1]
                
                batch_psnr = calculate_psnr(dehazed_norm, gt_norm)
                
                total_psnr += batch_psnr.item()
                
                if i == 0:  # Only visualize the first batch
                    visualize(hazy[:4], gt[:4], dehazed[:4], epoch, batch_psnr.item())
        
        avg_val_loss = val_loss / len(val_loader)
        avg_psnr = total_psnr / len(val_loader)
        
        val_losses.append(avg_val_loss)
        val_psnrs.append(avg_psnr)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, PSNR: {avg_psnr:.2f} dB")
        
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f"model_checkpoints/gunet_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'psnr': avg_psnr
            }, checkpoint_path)
            print(f"Model checkpoint saved at: {checkpoint_path}")
            
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'psnr': avg_psnr
            }, "model_checkpoints/gunet_best_model.pth")
            print(f"New best model saved with PSNR: {avg_psnr:.2f} dB")
    
    plot_metrics(train_losses, val_losses, val_psnrs, epochs)
    
    return train_losses, val_losses, val_psnrs

def visualize(hazy, gt, dehazed, epoch, psnr):
    hazy = hazy.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5
    gt = gt.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5
    dehazed = dehazed.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5
    
    plt.figure(figsize=(16, 12))
    for i in range(4):
        plt.subplot(3, 4, i+1)
        plt.imshow(np.clip(hazy[i], 0, 1))
        plt.title("Hazy")
        plt.axis("off")
        
        plt.subplot(3, 4, i+5)
        plt.imshow(np.clip(gt[i], 0, 1))
        plt.title("Ground Truth")
        plt.axis("off")
        
        plt.subplot(3, 4, i+9)
        plt.imshow(np.clip(dehazed[i], 0, 1))
        plt.title("Dehazed")
        plt.axis("off")
    
    plt.suptitle(f"Epoch {epoch+1} - PSNR: {psnr:.2f} dB")
    plt.tight_layout()
    plt.savefig(f"model_checkpoints/visualization_epoch_{epoch+1}.png")
    plt.close()

def plot_metrics(train_losses, val_losses, psnrs, epochs):
    plt.figure(figsize=(15, 8))
  
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True) 
  
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), psnrs)
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR on Validation Set')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("model_checkpoints/training_metrics.png")
    plt.close()

model = gUNet().to(device)
diffusion = DiffusionModel()

train_losses, val_losses, val_psnrs = train(model, diffusion, train_loader, val_loader, epochs=50, save_interval=5)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    psnr = checkpoint['psnr']
    
    print(f"Loaded checkpoint from epoch {epoch} with PSNR: {psnr:.2f} dB")
    return model, optimizer, epoch, train_loss, val_loss, psnr
