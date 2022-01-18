# Introduction to Computer Vision Project

![example](example.jpg)

## Files in this project

[`project.ipynb`](project.ipynb) - main file, contains all the stages in the project- preprocessing, generating more data, training and evaluating the model

[`predict.ipynb`](predict.ipynb) - file for getting prediction and results from the final model

[`preprocessing.py`](preprocessing.py) - this file create the dataset, or any dataset, from a given h5 file, outputted by the [SynthText](https://github.com/ankush-me/SynthText) project. It does so by cropping the letters, rotating them, and masking them with the average color of the letter.
It also has an option to cache the results for future runs. See more info at the file.

[`results.py`](results.py) - this file contains all the methods for evaluating the model, it contains methods for drawing the model accuracy graph, roc and matrix, it also has a method to output the predictions to csv.

[`vote.py`](vote.py) - this file contains the methods for the voting mechanism, fixing the predictions by voting between all the words letter.

[`config.py`](config.py) - this file contains the config file for the entire project, you can control multiple parameters via the [`config.ini`](config.ini) file, like: image size, the classes and more

[`plogging.py`](plogging.py) - this file contains a simple logger for the project

### Unused files in the project

[`augment.py`](augment.py) - this file was the first attempt to augment the images in the project, in the end I used tensorflow `ImageDataGenerator` to augment the images
[`generate.py`](generate.py) - this file was my first attempt to generate more images for the test data, in the end I did not use this file and used the given project with the relevant changes - https://github.com/yuvalshi0/SynthText/tree/python3

## Dependencies

**Python Version:** Python 3.9.9  
**Requirements:** tensorflow, pandas, numpy, sklearn, pillow, h5py, matplotlib

## Running prediction\training

### Get predictions from final model

- Place "final_model.h5" on this directory
- Final model can be found here: https://drive.google.com/drive/folders/1S25OyFxIbBq5HSBlJX0u6ujfpB_D0DVH?usp=sharing
- Remove cuda custom dll load (if needed)

All the prediction process can be found in the [`predict.ipynb`](predict.ipynb) notebook, just place the final model and the h5 file in the directory and run the notebook,
On my machine it takes about ~15 minutes

### Train the model

All the training and preprocess process can be found in the [`project.ipynb`](project.ipynb) file, to train the model just run the whole notebook,
Train time is about ~2 hours locally on my machine

#### Steps that might be needed:

- Configure the `config.ini` file with the correct paths (to the h5 file, for example)
- Remove the cuda dll os load (`os.add_dll_directory` used after the os import on both files), with this lines the GPU on my local computer didn't work

#### Code can also be seen at:

https://github.com/yuvalshi0/introduction-to-cv-course/tree/final-project
