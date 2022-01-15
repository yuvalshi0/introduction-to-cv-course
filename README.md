# Introduction to Computer Vision Project

## Files in this project

[`project.ipynb`](project.ipynb) - main file, contains all the stages in the project- preprocessing, generating more data, training and evaluating the model

[`preprocessing.py`](preprocessing.py) - this file create the dataset, or any dataset, from a given h5 file, outputted by the [SynthText](https://github.com/ankush-me/SynthText) project. It does so by cropping the letters, rotating them, and masking them with the average color of the letter.
It also has an option to cache the results for future runs. See more info at the file.

[`results.py`](results.py) - this file contains all the methods for evaluating the model, it contains methods for drawing the model accuracy graph, roc and matrix, it also has a method to output the predictions to csv.

[`vote.py`](vote.py) - this file contains the methods for the voting mechanism, fixing the predictions by voting between all the words letter.

[`config.py`](config.py) - TODO

[`plogging.py`](plogging.py) - TODO

### Unused files in the project

[`augment.py`](augment.py) - TODO  
[`generate.py`](generate.py) - TODO  

## To Run

TODO

> remove the cude file
> change config.ini file

## To predict
