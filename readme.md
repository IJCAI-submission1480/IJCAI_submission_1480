# IJCAI_submission_1480

***

### Requirements

* python 3
* pytorch 
* opencv-python

***

## Usage

The structure of our proposed network can be viewed in `model.py`. OCBlock and its stacked lightweight backbone are in `backbone.py`.

***

### Results

The matting results on Adobe Composition-1k testing set can be download [here](https://drive.google.com/file/d/1RDj47SxBpv45BKAMvlvHcZOtkl6-QJVA/view?usp=sharing). 
To calculate the metrics, you can use the code in [matlab_evaluation](./matlab_evaluation).

After download our results, you can just run [evaluation.m](./matlab_evaluation/evaluate.m) to get the **SAD, MSE, Grad, and Conn**.

You should complete the following settings:

* Set the ``trimap_path`` to the folder containing 1000 trimaps for test images

* Set the ``pred_path`` to the path of our results

* Set the ``alpha_path`` to the folder containing 50 ground truth images

Finally, run the script to get the metrics. 