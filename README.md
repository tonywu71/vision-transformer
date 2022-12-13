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

4. To generate the figures from the report, run



## Acknowledgments

Note that our VIT architecture is following the one from *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Dosovitskiy, 2021*. I also used some lines of codes from the [Keras website](https://keras.io/examples/vision/image_classification_with_vision_transformer/).
