import torch
from torch import nn

### WARNING: DO NOT EDIT THE CLASS NAME, INITIALIZER, AND GIVEN INPUTS AND ATTRIBUTES. OTHERWISE, YOUR TEST CASES CAN FAIL. ###


class DCGAN(nn.Module):
    def __init__(self, image_channels=3, latent_dim=128, base_channels=64):
        super().__init__()
        self.image_channels = image_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.generator = self._build_generator(latent_dim, base_channels, image_channels)
        self.discriminator = self._build_discriminator(base_channels, image_channels)
        self.criterion = nn.BCEWithLogitsLoss()

    def _make_gen_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        return block

    def _make_disc_block(self, in_channels, out_channels, kernel_size, stride, padding, use_batchnorm):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        block = nn.Sequential(*layers)

        return block

    def _build_generator(self, latent_dim, base_channels, image_channels):
        specs = [
            (latent_dim, base_channels * 4, 4, 1, 0),
            (base_channels * 4, base_channels * 2, 4, 2, 1),
            (base_channels * 2, base_channels, 4, 2, 1),
        ]
        blocks = []
        for in_channels, out_channels, kernel, stride, padding in specs:
            blocks.append(self._make_gen_block(in_channels, out_channels, kernel, stride, padding))
        blocks.append(nn.ConvTranspose2d(base_channels, image_channels, 4, 2, 1))
        blocks.append(nn.Tanh())
        generator = nn.Sequential(*blocks)
        return generator

    def _build_discriminator(self, base_channels, image_channels):
        specs = [
            (image_channels, base_channels, 4, 2, 1, False),
            (base_channels, base_channels * 2, 4, 2, 1, True),
            (base_channels * 2, base_channels * 4, 4, 2, 1, True),
        ]
        blocks = []
        for in_channels, out_channels, kernel, stride, padding, use_bn in specs:
            blocks.append(self._make_disc_block(in_channels, out_channels, kernel, stride, padding, use_bn))
        blocks.append(nn.Conv2d(base_channels * 4, 1, 4, 1, 0))
        discriminator = nn.Sequential(*blocks)
        return discriminator

    def sample(self, z):
        if z.ndim == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.generator(z)

    def forward(self, batch):
        real = batch["images"]
        batch_size = real.size(0)
        device = real.device
        noise = batch.get("noise")
        if noise is None:
            noise = noise = torch.randn(batch_size, self.latent_dim, device=device)  # NOTE: draw latent noise for the generator
        
        
        fake_images = self.generator(noise.view(batch_size, self.latent_dim, 1, 1))  # NOTE: generate fake images from the latent noise
        
        
        logits_real = self.discriminator(real)  # NOTE: score real images with the discriminator
        logits_fake_detached = self.discriminator(fake_images.detach())  # NOTE: score fake images without backpropagating into G
        
        ones = torch.ones_like(logits_real)  # targets for real samples
        zeros = torch.zeros_like(logits_fake_detached)  # targets for fake samples
        
        loss_d_real = self.criterion(logits_real, ones)
        loss_d_fake = self.criterion(logits_fake_detached, zeros)
        discriminator_loss = (loss_d_real + loss_d_fake)  # NOTE: compute the discriminator loss 
        logits_fake = self.discriminator(fake_images)  # NOTE: score fake images for the generator update
        generator_loss = self.criterion(logits_fake, ones)  # NOTE: compute the generator loss
        loss = discriminator_loss + generator_loss
        return {
            "loss": loss,
            "generator_loss": generator_loss,
            "discriminator_loss": discriminator_loss,
            "fake_images": fake_images,
        }