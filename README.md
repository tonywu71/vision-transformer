# vision-transformer
Tensorflow implementation of Image Classification with Vision Transformer on the MNIST dataset.



## Instructions

1. Using an environment with `python 3.10.8`, install modules using:

   ```bash
   pip install -r requirements.txt
   ```

2. To train and evaluate the VIT model, run:

   ```bash
   python train_VIT.py
   ```

3. To train and evaluate the CNN model (benchmark model), run:

   ```bash
   python train_CNN.py
   ```

4. To generate the figures from the report, run:

   ```bash
   python experiments/impact_of_ds_size.py
   ```



## Acknowledgments

Note that our VIT architecture is the same one as the one presented in *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Dosovitskiy, 2021*. Some lines of codes were taken from the [Keras tutorial on "Image classification with Vision Transformer"](https://keras.io/examples/vision/image_classification_with_vision_transformer/).
