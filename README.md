**Notes:**


+ For model training, the CelebA dataset was utilized. This dataset offers more than 200K facial images of global celebrities, each of which consist of 40 unique attributes (labels). The complete dataset and labels files can be accessed and imported from Kaggle at the following link: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset 

+ The _saved_model_pc_ and _saved_model_spc_ files contain model results achieved from fine-tuning different primitives. While the former is for Perturbation and 2D Convolution filters, the latter has additional Shift operation during training.

+ The model was trained on CUDA and NVIDIA RTX A2000, resulting each epoch taking three hours to be done. Also, due to fast convergence of training accuracy, the training was interrupted to save the model for future training.


**Summary:**

For the sake of research and academic learning, this project is best used as a reference for other future model architectures.  
