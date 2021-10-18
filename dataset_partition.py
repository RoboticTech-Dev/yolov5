import os 
import shutil
import subprocess
from sklearn.model_selection import train_test_split

#python train.py --img 416 --batch 16 --epochs 100 --data '.\data\Deode_data.yaml' --cfg ./models/yolov5l_Deode.yaml --name yolov5_Deode_results  --cache
#python detect.py --weights runs/train/yolov5_Deode_results4/weights/best.pt --img 416 --conf 0.4 --source 'C:\Users\matha\Documents\Github\Deode_Dataset\images\test'

IMG_PATH   = r'C:/Users/matha/Documents/Github/Deode_Dataset/images'
LABEL_PATH = r'C:/Users/matha/Documents/Github/Deode_Dataset/labels'

# Trainning Parameters
CLASSES_YAML = './data/Deode_data.yaml'
MODEL_YAML   = './models/yolov5l_Deode.yaml'

#Detecting parameters
WEIGHTS_PATH = "./runs/train/yolov5_Deode_results4/weights/best.pt"
IMG_TEST_PATH = "D:/GoogleDrive/ROBOTICTECH/PROJETOS/Deode/Fotos/Fotos_Reatores"

# Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False


# Read images and labels
images = [os.path.join(IMG_PATH, x) for x in os.listdir(IMG_PATH) if x[-3:] == 'jpg' or x[-3:] == 'JPG']
annotations = [os.path.join(LABEL_PATH, x) for x in os.listdir(LABEL_PATH) if x[-3:] == "txt"]

images.sort()       # Sorts the images path
annotations.sort()  # Sorts the label path

if images:
    
    # Split the dataset into train-valid-test splits  
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

    # Move the splits into their folders
    move_files_to_folder(train_images, IMG_PATH +'/train')
    move_files_to_folder(val_images, IMG_PATH+'/val/')
    move_files_to_folder(test_images, IMG_PATH+'/test/')
    move_files_to_folder(train_annotations, LABEL_PATH+'/train/')
    move_files_to_folder(val_annotations, LABEL_PATH+'/val/')
    move_files_to_folder(test_annotations, LABEL_PATH+'/test/')

# Trainning
#subprocess.call("python train.py --img 416 --batch 16 --epochs 100 --data "+CLASSES_YAML+" --cfg "+MODEL_YAML+" --name yolov5_Deode_results  --cache",shell=True)

# Detecting
subprocess.call("python detect.py --weights "+WEIGHTS_PATH+" --img 416 --conf 0.4 --source "+IMG_TEST_PATH)