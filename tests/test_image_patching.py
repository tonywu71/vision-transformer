from keras.datasets.cifar100 import load_data
from utils.plot import plot_patches

def test_plot_patches():
    # ------ Hyperparameters ------
    patch_size = 6  # Size of the patches to be extract from the input images
    image_size = 72  # We'll resize input images to this size
    
    # ------ Load dataset ------
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # ------ Plot a random patch from the train set ------
    try:
        plot_patches(x_train, patch_size=patch_size, image_size=image_size)
    except:
        raise
    
    return
