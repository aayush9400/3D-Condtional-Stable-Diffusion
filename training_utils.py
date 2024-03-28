import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import wandb

class WandbImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, val_dataset, layer_idx=64, num_images=2, log_freq=10):
        # Initialize with the VQVAE model and validation dataset
        self.model = model
        self.val_dataset = val_dataset
        self.layer_idx = layer_idx  # The specific layer to extract and log
        self.num_images = num_images
        self.log_freq = log_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_freq == 0:
            # Generate a batch from the validation dataset
            for x, mask, _ in self.val_dataset.take(self.num_images):
                break
            x = tf.concat([x, mask], axis=-1)
            # Ensure we only use a subset for visualization to save resources
            x_subset = x[:self.num_images]

            # Get the reconstructions from the VQVAE model
            reconstructions, _ = self.model(x_subset, training=False)
            img_reconstructed, mask_reconstructed = tf.split(reconstructions, num_or_size_splits=2, axis=-1)
            # print(x.shape, type(x), img_reconstructed.shape, type(img_reconstructed))
            # Prepare images for logging
            images = []
            for i in range(self.num_images):
                # Extract the specific layer from the original and reconstructed images
                original_slice = x[i, :, :, self.layer_idx, 0].numpy()
                reconstructed_slice = img_reconstructed[i,:,:,self.layer_idx,0].numpy()
                # print(original_slice.shape, type(original_slice), reconstructed_slice.shape, type(reconstructed_slice))
                # Normalize and convert to uint8
                original_slice = np.clip(original_slice * 255.0, 0, 255).astype(np.uint8)
                reconstructed_slice = np.clip(reconstructed_slice * 255.0, 0, 255).astype(np.uint8)

                # Plot original and reconstructed slices side by side
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(original_slice, cmap='gray')
                axes[0].set_title('Original')
                axes[0].axis('off')
                axes[1].imshow(reconstructed_slice, cmap='gray')
                axes[1].set_title('Reconstructed')
                axes[1].axis('off')
                plt.tight_layout()

                # Convert the plot to an image array
                fig.canvas.draw()
                img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                # Close the plot to free memory
                plt.close(fig)

                # Append the side-by-side image to the list for logging
                images.append(wandb.Image(img_array, caption=f"Epoch {epoch+1} Original & Reconstructed"))

            # Log the images to W&B
            wandb.log({"Original & Reconstructions": images}, commit=False)