import os
import math
from sklearn import neighbors
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
def training(train_dir,model_save_location,n_neighbors=2,knn_algo='ball_tree',verbose=False):
    x=[]
    y=[]
    #We will loop through each folder in training directory and check each image in folders
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir,class_dir)):
            continue
        #Looping through each image file for every person
        for img_path in image_files_in_folder(os.path.join(train_dir,class_dir)):
            image= face_recognition.load_image_file(img_path)
            face_count=face_recognition.face_locations(image)
            print("Curretly on Image: ",img_path)
            if len(face_count)!=1:
                #If there are more than one or no people in Image we can't use for training
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't found a face" if len(face_count)< 1 else "Found too many faces"))
            else:
                #If every thing is right for image then we add it to the training set
                x.append(face_recognition.face_encodings(image,known_face_locations=face_count)[0])
                y.append(class_dir)
    #Figuring the number of neighbors we should use for weighting
    if n_neighbors is None:
        n_neighbors= int(round(math.sqrt(len(x))))
        if verbose:
            print("Chosen n_neighbors automatically",n_neighbors)
    knn_clf= neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,algorithm=knn_algo,weights='distance')
    knn_clf.fit(x,y)
    #Saving the trained model
    if model_save_location is not None:
        with open(model_save_location,'wb') as f:
            pickle.dump(knn_clf,f)
        print("Training is complete your trained model has been saved to the given location!!")
    return knn_clf

training("training_image","Trained_model/trained_knn_model.clf")

