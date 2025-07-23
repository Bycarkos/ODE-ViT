import torch
import torchvision.transforms.v2.functional as TF
import  torchvision.transforms.v2 as transforms
import random


class KoopmanCropTransform:
    def __init__(self, crop_width: int = 64, shift: int = 8):
        self.crop_width = crop_width
        self.shift = shift

    def __call__(self, image):
        """
        Args:
            image (Tensor): (C, H, W) format
        Returns:
            g0, g1: two crops shifted by `shift` pixels along the width
        """
        C, H, W = image.shape
        max_start = W - self.crop_width - self.shift

        if max_start <= 0:
            raise ValueError(f"Image width {W} too small for crop + shift.")

        # Random start location
        start_x = random.randint(0, max_start)
        g0 = image[:, :, start_x : start_x + self.crop_width]
        g1 = image[:, :, start_x + self.shift : start_x + self.shift + self.crop_width]

        return g0, g1
    
    


if __name__ == "__main__":
        # Example usage
    from PIL import Image
    import matplotlib.pyplot as plt
    import requests
    from io import BytesIO
    url = "/data/users/cboned/data/HTR/Esposalles/IEHHR_training_part1/idPage10354_Record1/words/idPage10354_Record1_Line0_Word0.png"
    image = Image.open(url).convert("RGB")

    # Convert to tensor and normalize to [0,1]
    image = TF.to_image(image)
    image = TF.to_dtype(image, torch.float32, scale=True)
    augment = transforms.Compose([
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.75),
        transforms.RandomApply([transforms.RandomAffine(
            degrees=5,  # small rotation
            translate=(0.15, 0.25),  # slight shift
            scale=(0.95, 1.15),  # small zoom
            shear=5)], p=1),
        transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=5)], p=0.5),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=0.3, contrast=0.3)], p=0.5),
    ])
    image_aug = augment(image)
    transform = KoopmanCropTransform(crop_width=128, shift=8)
    g0, g1 = transform(image)
    
        # Plot original and cropped versions
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    print(image.permute(1, 2, 0).shape)
    axes[0].imshow(image.permute(1, 2, 0).numpy())
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(image_aug.permute(1, 2, 0).numpy())
    axes[1].set_title("Original Image")
    axes[1].axis("off")

    axes[2].imshow(g0.permute(1, 2, 0).numpy())
    axes[2].set_title("Crop g₀ (t)")
    axes[2].axis("off")

    axes[3].imshow(g1.permute(1, 2, 0).numpy())
    axes[3].set_title("Crop g₁ (t+Δt)")
    axes[3].axis("off")

    plt.tight_layout()
    output_path = "./koopman_crops_example.png"
    plt.savefig(output_path)
    print(g0.shape, g1.shape)  # Should print (3, 64, 64) for both