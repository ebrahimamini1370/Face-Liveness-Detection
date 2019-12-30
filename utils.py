# import necessary packages:
import time
import os
import numpy as np
from skimage import feature
from PIL import Image, ImageDraw, ImageFont
from imutils import paths
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score #calculate accuracy

#function for showing texts in images, from a language other than English
def draw_ch_zn(image,str_txt,font_path,location):
    font = ImageFont.truetype(font_path,20,encoding='utf-8')
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    draw.text((location[0],location[1]),str_txt,(255,0,0),font)
    return np.array(pil_image).copy()

# function for enhancing images, according to the paper
def enhance_face(image):

    # resize image
    newsize = (64, 64)
    image = image.resize(newsize, Image.ANTIALIAS)

    image_arr = np.array(image)
    image_arr = image_arr.ravel()
    image_list = list(image_arr)

    f_x = {}
    for i in range(256):
        f_x[i] = image_list.count(i)

    F_x = {}
    G_x = {}
    for i in range(256):
        F_x[i] = 0
        for j in range(i + 1):
            F_x[i] += f_x[j]
        G_x[i] = F_x[i] / (64*64)
        G_x[i] *= 256

    new_image_list = []
    for i in range(len(image_list)):
        new_image_list.append(G_x[image_list[i]])

    new_image_arr = np.array(new_image_list).reshape(64, 64)

    return new_image_arr

# function for compute lbp from enhanced images
def lbp_extraction(image, number_points, radius, eps=1e-7):
    # enhance the image:
    image = enhance_face(image)

    # compute the Local Binary Pattern features
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = feature.local_binary_pattern(image,number_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, number_points + 3),
                             range=(0, number_points + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    # return the histogram of Local Binary Patterns
    return hist

# function for converting all images in database, to a database including
# lbpatterns and labels
def preparing_dataset_images(number_points, radius):
    labels = []
    data = []
    # loop over the training images
    for imagePath in paths.list_images('Dataset'):
        print(imagePath)
        # load the image, convert it to grayscale
        image = Image.open(imagePath).convert('L')

        hist = lbp_extraction(image, number_points, radius)

        # extract the label from the image path, then update the
        # label and data lists
        data.append(hist)
        labels.append(imagePath.split(os.path.sep)[1])

    return (data, labels)

# function for converting all images to thier corresponding lbps and labels,
# and providing a database appropriate for binary classification tasks,
# and then, save the resulting database
def save_liveness_detection_database(number_points, radius):
    start_preparing_database = time.time()
    database = preparing_dataset_images(number_points, radius)
    end_preparing_database = time.time()
    with open('liveness_detection_database.pkl', 'wb') as outfile:
        pickle.dump(database, outfile)
    print(f"Preparing database was done in {end_preparing_database - start_preparing_database} seconds")
    
# function for loading database for implementing classification tasks
def load_liveness_detection_database():
    with open('liveness_detection_database.pkl', 'rb') as infile:
        database = pickle.load(infile)
    return database


# function for train a svm classifier, and save the classifier
def train_svm_and_save_it():
    data, labels = load_liveness_detection_database()
    data_train, data_test, label_train, label_test = train_test_split(data, labels)
    # we define the classifier by model, maybe I want to use another type classifiers in future
    model = SVC(kernel="rbf", gamma = 'auto', C=0.001)
    start_training_time = time.time()
    model.fit(data_train, label_train)
    end_training_time = time.time()
    print(f"Training the model was done in {end_training_time - start_training_time} seconds")

    with open('svm_clf.pkl', 'wb') as outfile:
        pickle.dump(model, outfile)

    predicted = model.predict(data_test)
    model_accuracy = accuracy_score(label_test, predicted)
    print(f"Accuracy is {model_accuracy}")
# function for loading the classifier
def load_svm_clf():
    with open('svm_clf.pkl', 'rb') as infile:
        model = pickle.load(infile)
    return model

# function for loading classifier and use it for liveness prediction
def predict_liveness(test_imge, number_points, radius):
    model = load_svm_clf()
    hist = lbp_extraction(test_imge, number_points, radius).reshape(1, -1)
    return model.predict(hist)

#function for doing liveness detection real-time, it does not load the classifier for every frame
def predict_liveness_real_time(test_imge, model, number_points, radius):
    hist = lbp_extraction(test_imge, number_points, radius).reshape(1, -1)
    return model.predict(hist)
