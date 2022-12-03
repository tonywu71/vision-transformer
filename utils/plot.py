import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from preprocessing.image_patching import Patches


def plot_patches(x_train: tf.Tensor, patch_size: int, image_size: int) -> None:
    """Note: works with square images only."""
    
    n_channels = x_train.shape[-1]
    
    plt.figure(figsize=(4, 4))
    image = x_train[np.random.choice(range(x_train.shape[0]))] # type: ignore
    
    if n_channels==1:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image.astype("uint8"))
    plt.axis("off")

    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]), size=(image_size, image_size)
    )
    patches = Patches(patch_size)(resized_image)
    print(f"Image size: {image_size} X {image_size}")
    print(f"Patch size: {patch_size} X {patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, n_channels))
        if n_channels==1:
            plt.imshow(patch_img.numpy(), cmap="gray")
        else:
            plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")
    
    return
