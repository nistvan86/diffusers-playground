import numpy as np
import torch
from diffusers import ZImagePipeline


def calculate_coefficients():
    print("Loading Z-Image-Turbo pipeline...")
    try:
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.float16,
        )
        pipe.to("cuda")
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        return

    vae = pipe.vae
    vae.eval()

    # Get config values
    shift = vae.config.shift_factor if hasattr(vae.config, "shift_factor") else 0.0
    scale = vae.config.scaling_factor if hasattr(vae.config, "scaling_factor") else 1.0

    print(f"Shift: {shift}, Scale: {scale}")

    # Generate random latents
    # We want to cover the typical range of latents.
    # Diffusion latents are typically N(0, 1) or close to it.
    num_samples = 2000
    latent_channels = vae.config.latent_channels

    # Create random latents representing the input to the preview function
    # These mimic the latents during diffusion
    latents_in = torch.randn(
        num_samples, latent_channels, 1, 1, device="cuda", dtype=torch.float16
    )

    # Process latents for VAE decoding as done in main.py
    latents_for_vae = latents_in.clone()
    if shift is not None:
        latents_for_vae = latents_for_vae - shift
    if scale is not None:
        latents_for_vae = latents_for_vae / scale

    latents_for_vae = latents_for_vae.to(dtype=vae.dtype)

    print("Decoding latents (this might take a moment)...")
    with torch.no_grad():
        # Decode
        # VAE decode expects [B, C, H, W]
        decoded_images = vae.decode(latents_for_vae, return_dict=False)[0]

    # decoded_images is [N, 3, H, W] (here H=W=8 due to 1x1 latent and downscale 8?)
    # Wait, SDXL VAE downscale factor is 8.
    # If input is 1x1, output should be 8x8.
    # But we want the average color or just the relationship.
    # Actually, we can use 1x1 latents and see what comes out.
    # Ideally we want to map pixel-wise.
    # Latent[x, y] -> RGB[x*8, y*8] ... RGB[x*8+7, y*8+7]
    # But Latent2RGB usually maps one latent vector to one RGB vector (or a patch).
    # Typically it maps to the *spatial average* of the decoded patch, or similar.

    # Let's check the output size
    print(f"Decoded shape: {decoded_images.shape}")

    # We want to map latent vector (16,) to RGB vector (3,).
    # We can average the 8x8 pixels to get a single RGB value for that latent.
    # Or we can just use the center pixel. Averaging is probably more robust.

    # Clamp and scale to [0, 1] for regression
    decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)

    # Mean over spatial dimensions
    # [N, 3, H, W] -> [N, 3]
    rgb_values = decoded_images.mean(dim=[2, 3])

    # Prepare data for regression
    # X: [N, 16] (latents_in)
    # Y: [N, 3] (rgb_values)

    X = latents_in.view(num_samples, latent_channels).float()
    Y = rgb_values.view(num_samples, 3).float()

    # Add bias term to X
    X_bias = torch.cat([X, torch.ones(num_samples, 1, device="cuda")], dim=1)

    # Solve A @ W = B
    # W = (A^T A)^-1 A^T B
    # or using lstsq

    print("Solving for coefficients...")
    result = torch.linalg.lstsq(X_bias, Y)
    weights = result.solution  # [17, 3]

    # The weights matrix W is such that X_bias @ W approx Y
    # W has shape [17, 3].
    # The first 16 rows are the coefficients for the latent channels.
    # The last row is the bias.

    w_coefs = weights[:-1, :]  # [16, 3]
    bias = weights[-1, :]  # [3]

    print("\n--- Calculated Coefficients ---")
    print("Latent -> RGB Matrix (transpose of W_coefs for easy reading):")
    # Usually we represent it as:
    # R = c1*L1 + ... + bias

    print("COEFFICIENTS = [")
    for i in range(latent_channels):
        row = w_coefs[i].cpu().numpy()
        print(f"    [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}], # Latent channel {i}")
    print("]")

    print("BIAS = [")
    b = bias.cpu().numpy()
    print(f"    {b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}")
    print("]")

    # Validation
    print("\nValidation on random sample:")
    pred = X_bias[0] @ weights
    actual = Y[0]
    print(f"Predicted: {pred.cpu().numpy()}")
    print(f"Actual:    {actual.cpu().numpy()}")


if __name__ == "__main__":
    calculate_coefficients()
