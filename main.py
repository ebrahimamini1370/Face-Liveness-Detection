from utils import *

# initializing parameters
number_points = 8
radius = 2

#call this function for providing and saving appropriate data in dataset
save_liveness_detection_database(number_points, radius)

#call this function for training a classifier on dataset
train_svm_and_save_it()
