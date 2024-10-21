import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch.utils.data.dataset import Dataset
import gan_utils as gan_utils

# Constants and Hyperparameters
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 6
NOISE_DIM = 60
BATCH_SIZE = 512
EPOCHS = 500
LR = 0.0002
BETA1 = 0.5  # Beta1 hyperparam for Adam optimizers

# Check for GPU availability
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Placeholder for the dataset
# Replace this with your data loading logic
# Ensure your data is normalized to [-1, 1] range
# dataset = YourDatasetClass()
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: (batch_size, 60, 1, 1)
            nn.ConvTranspose2d(60, 1024, kernel_size=4, stride=1, padding=0),  # Output: (1024, 4, 4)
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # Additional layer
            nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=1, padding=1),  # Output: (1024, 4, 4)
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),   # Output: (512, 8, 8)
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # Additional layer
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),    # Output: (512, 8, 8)
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),    # Output: (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Additional layer
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),    # Output: (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),    # Output: (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Additional layer
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),    # Output: (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),     # Output: (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Additional layer
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),      # Output: (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, IMG_CHANNELS, kernel_size=3, stride=1, padding=1),  # Output: (IMG_CHANNELS, 64, 64)
            nn.Tanh()  # Output values between -1 and 1
        )

    def forward(self, x):
        return self.main(x)


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: (batch_size, IMG_CHANNELS, 64, 64)
            nn.Conv2d(IMG_CHANNELS, 64, kernel_size=4, stride=2, padding=1),   # Output: (64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),

            # Additional layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),             # Output: (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),            # Output: (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Additional layer
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),           # Output: (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),           # Output: (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Additional layer
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),           # Output: (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),           # Output: (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Additional layer
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),           # Output: (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),             # Output: (1, 1, 1)
            nn.Sigmoid()  # Use sigmoid if you're applying BCEWithLogitsLoss
        )

    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1)


def main():
    folder_name = r'T:\600\60010\FakeImageDataset\Distinguish\test'
    dataset = \
        gan_utils.CustomDatasetFromGRD(folder_name, 64, 64, transforms=None,
                                     channels=6,
                                     constant_axis=1,
                                     do_flip=False,
                                     porous=6,
                                     max_files=100,
                                     max_samples=100000,
                                     stride_x=32,
                                     stride_y=32)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512,
                                             shuffle=True, num_workers=2)

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Loss Function
    criterion = nn.BCELoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))

    # Training Loop
    for epoch in range(EPOCHS):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update Discriminator
            ############################
            ## Train with real images
            discriminator.zero_grad()
            real_images = data[0].to(device)  # Assuming dataset returns (data, labels)
            b_size = real_images.size(0)
            label = torch.full((b_size,), 1.0, dtype=torch.float, device=device)  # Real label = 1

            output = discriminator(real_images).view(-1)
            loss_D_real = criterion(output, label)
            loss_D_real.backward()

            ## Train with fake images
            noise = torch.randn(b_size, NOISE_DIM, device=device)
            fake_images = generator(noise)
            label.fill_(0.0)  # Fake label = 0

            output = discriminator(fake_images.detach()).view(-1)
            loss_D_fake = criterion(output, label)
            loss_D_fake.backward()

            # Update Discriminator
            loss_D = loss_D_real + loss_D_fake
            optimizer_D.step()

            ############################
            # (2) Update Generator
            ############################
            generator.zero_grad()
            label.fill_(1.0)  # Generator tries to make discriminator believe the fake images are real

            output = discriminator(fake_images).view(-1)
            loss_G = criterion(output, label)
            loss_G.backward()

            # Update Generator
            optimizer_G.step()

            # Print training stats
            if i % 50 == 0:
                print(
                    f'Epoch [{epoch+1}/{EPOCHS}] Batch {i}/{len(dataloader)} \
                    Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}'
                )

        # Optional: Save model checkpoints or generate sample images
        print(f'Epoch {epoch + 1}/{EPOCHS} completed.')

    # Save the final models
    # torch.save(generator.state_dict(), 'generator.pth')
    # torch.save(discriminator.state_dict(), 'discriminator.pth')


if __name__ == '__main__':
    main()
