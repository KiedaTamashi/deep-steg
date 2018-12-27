import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.misc
from keras.preprocessing import image
import os
import subprocess
import easygui

DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

IMG_SHAPE = (64, 64)



def pixel_errors(input_S, input_C, decoded_S, decoded_C):
    """Calculates mean of Sum of Squared Errors per pixel for cover and secret images. """
    see_Spixel = np.sqrt(np.mean(np.square(255 * (input_S - decoded_S))))
    see_Cpixel = np.sqrt(np.mean(np.square(255 * (input_C - decoded_C))))

    return see_Spixel, see_Cpixel

# TODO debug
def pixel_histogram(diff_S, diff_C):
    """Calculates histograms of errors for cover and secret image. """
    diff_Sflat = diff_S.flatten()
    diff_Cflat = diff_C.flatten()

    fig = plt.figure(figsize=(15, 5))
    a = fig.add_subplot(1, 2, 1)

    imgplot = plt.hist(255 * diff_Cflat, 100, normed=1, alpha=0.75, facecolor='red')
    a.set_title('Distribution of error in the Cover image.')
    plt.axis([0, 250, 0, 0.2])

    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.hist(255 * diff_Sflat, 100, normed=1, alpha=0.75, facSecolor='red')
    a.set_title('Distribution of errors in the Secret image.')
    plt.axis([0, 250, 0, 0.2])

    plt.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def show_image(img, n_rows, n_col, idx, gray=False, first_row=False, title=None):
    ax = plt.subplot(n_rows, n_col, idx)
    if gray:
        plt.imshow(rgb2gray(img), cmap = plt.get_cmap('gray'))
    else:
        plt.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if first_row:
        plt.title(title)

# Configs for results display
def result_display(input_S,input_C,decoded_S,decoded_C,
                   SHOW_GRAY = False,SHOW_DIFF = True,ENHANCE = 1,n = 6):
    '''

    :param SHOW_GRAY: Show images in gray scale
    :param SHOW_DIFF: Show difference bettwen predictions and ground truth.
    :param ENHANCE: Diff enhance magnitude
    :param n: Number of secret and cover pairs to show.
    :return:
    '''
    # Get absolute difference between the outputs and the expected values.
    diff_S, diff_C = np.abs(decoded_S - input_S), np.abs(decoded_C - input_C)

    # Print pixel-wise average errors in a 256 scale.
    S_error, C_error = pixel_errors(input_S, input_C, decoded_S, decoded_C)

    print("S error per pixel [0, 255]:", S_error)
    print("C error per pixel [0, 255]:", C_error)
    # pixel_histogram(diff_S, diff_C)


    plt.figure(figsize=(14, 15))
    rand_indx = [random.randint(0, 1000) for x in range(n)]
    # for i, idx in enumerate(range(0, n)):
    for i, idx in enumerate(rand_indx):
        n_col = 6 if SHOW_DIFF else 4

        show_image(input_C[idx], n, n_col, i * n_col + 1, gray=SHOW_GRAY, first_row=i == 0, title='Cover')

        show_image(input_S[idx], n, n_col, i * n_col + 2, gray=SHOW_GRAY, first_row=i == 0, title='Secret')

        show_image(decoded_C[idx], n, n_col, i * n_col + 3, gray=SHOW_GRAY, first_row=i == 0, title='Encoded Cover')

        show_image(decoded_S[idx], n, n_col, i * n_col + 4, gray=SHOW_GRAY, first_row=i == 0, title='Decoded Secret')

        if SHOW_DIFF:
            show_image(np.multiply(diff_C[idx], ENHANCE), n, n_col, i * n_col + 5, gray=SHOW_GRAY, first_row=i == 0,
                       title='Diff Cover')

            show_image(np.multiply(diff_S[idx], ENHANCE), n, n_col, i * n_col + 6, gray=SHOW_GRAY, first_row=i == 0,
                       title='Diff Secret')

    plt.show()


def load_dataset_small(num_images_per_class_train=10, num_images_test=500):
    """Loads training and test datasets, from Tiny ImageNet Visual Recogition Challenge.

    Arguments:
        num_images_per_class_train: number of images per class to load into training dataset.
        num_images_test: total number of images to load into training dataset.
    """
    X_train = []
    X_test = []

    # Get training image directory.
    path = easygui.diropenbox()

    # Create training set.
    for c in os.listdir(path):
        c_dir = os.path.join(path, c, 'images')
        c_imgs = os.listdir(c_dir)
        random.shuffle(c_imgs)
        for img_name_i in c_imgs[0:num_images_per_class_train]:
            img_i = image.load_img(os.path.join(c_dir, img_name_i))
            x = image.img_to_array(img_i)
            X_train.append(x)
    random.shuffle(X_train)

    # Create test set.
    test_dir = os.path.join(TEST_DIR, 'images')
    test_imgs = os.listdir(test_dir)
    random.shuffle(test_imgs)
    for img_name_i in test_imgs[0:num_images_test]:
        img_i = image.load_img(os.path.join(test_dir, img_name_i))
        x = image.img_to_array(img_i)
        X_test.append(x)

    # Return train and test data as numpy arrays.
    return np.array(X_train), np.array(X_test)

def ffmpegProcess(code):
    '''
    run ffmepg code
    '''
    getmp3 = code
    returnget = subprocess.call(getmp3,shell=True)
    print(returnget)

def extractFrameOfVideo(video_path,frame_rate=30,frame_save_path='./coverSource'):
    DivideCode = 'ffmpeg -i ' + video_path + ' -r '+str(frame_rate)+' '+frame_save_path+'%06d.png'
    ffmpegProcess(DivideCode)

def generateVideo(frame_save_path='./hideSource',output_path='./test.mp4',frame_rate=30):
    generateCode = "ffmpeg -framerate "+str(frame_rate)+" -i "+frame_save_path+"\%06d.png -vcodec libx264 -r "\
                   +str(frame_rate)+" -pix_fmt yuv420p "+output_path
    ffmpegProcess(generateCode)

def readFrames(file_path):
    '''
    :return: list of framePath and num of file
    '''
    fs = os.listdir(file_path)
    fs.sort(key=lambda x: int(x[:-4]))
    file_name_list = []
    cnt=0
    for f in fs:
        file_name_list.append(os.path.join(file_path,f))
        cnt += 1
    return file_name_list,cnt

def randomSort(file_name_list,length,key,mode='encode'):
    '''
    if you want to recover the length and key must keep same
    :param file_name_list:
    :param length: number of files
    :param key: as seed
    :return: resorted list
    '''

    random.seed(key)
    # generate the random order
    rs = random.sample(range(length),length)
    resorted_list = []
    if mode=='encode':
        for i in range(length):
            resorted_list.append(file_name_list[rs[i]])
        print(resorted_list)
    elif mode =='decode':
        tmp = list(range(length))
        for i in range(length):
            tmp[rs[i]] = file_name_list[i]
        resorted_list = tmp
        print(resorted_list)
    else:
        print('mode wrong\n')

    return resorted_list


a = list(range(10))
print(a)
b = randomSort(a,10,2)
randomSort(b,10,2,mode='decode')
b = randomSort(a,10,2)
randomSort(b,10,2,mode='decode')