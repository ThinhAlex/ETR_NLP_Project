**Notes:**


+ This model was trained on the CelebA dataset on 200K facial images of global celebrities. The dataset consists of 40 unique attributes (labels). The complete dataset and labels files can be accessed and imported from Kaggle at the following link: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset 

+ The _saved_model_pc_ and _saved_model_spc_ files contain model results achieved from fine-tuning the primitives parameter. The _saved_model_pc_ image displays training results when Perturbation and 2D Convolution filters were used. The _saved_model_spc_ Perturbation, 2D Convolution, and Shift.

+ The model was trained on CUDA and NVIDIA RTX A2000, and each epoch took three hours to complete. Also, due to the fast convergence of training accuracy, the training was interrupted to save the model for future training.
