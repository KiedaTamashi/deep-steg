from deepSteg import *

#just validation

def load_dataset_video(num_frames_theta=100):
    X_secret = []
    X_cover = []
    X_size = []


    # Get training dataset directory. It should contain 'train' folder and 'test' folder.
    path = easygui.diropenbox(title = 'Choose dataset directory')

    # Create test set.
    secret_dir = os.path.join(path, 'secret','image')
    cover_dir = os.path.join(path,'cover','image')
    secret_imgs = os.listdir(secret_dir)
    secret_imgs.sort(key=lambda x: int(x[:-4]))
    cover_imgs = os.listdir(cover_dir)
    cover_imgs.sort(key=lambda x: int(x[:-4]))
    if len(cover_imgs)<len(secret_imgs):
        print("\nError! num of frames\n")
    num_frames_test = len(secret_imgs)
    if num_frames_test>num_frames_theta:
        print("\nToo many frames\n")
        num_frames_test = num_frames_theta
    # must be same size
    for img_name_i in secret_imgs[0:num_frames_test]:
        img_i = image.load_img(os.path.join(secret_dir, img_name_i))
        #resize
        img_i_reshape,img_ori_size = resize_image(img_i)
        x = image.img_to_array(img_i_reshape)
        X_secret.append(x)
        X_size.append(img_ori_size)
    for img_name_i in cover_imgs[0:num_frames_test]:
        img_i = image.load_img(os.path.join(cover_dir, img_name_i))
        #resize
        img_i_reshape,img_ori_size = resize_image(img_i)
        x = image.img_to_array(img_i_reshape)
        X_cover.append(x)


    # Return train and test data as numpy arrays.
    return np.array(X_secret), np.array(X_cover), X_size

def load_frames_preprocess(num_images_theta=100):
    input_S_orig, input_C_orig, X_test_orig_size= load_dataset_video(num_images_theta)
    # Normalize image vectors.
    input_S = input_S_orig / 255.
    input_C = input_C_orig / 255.

    # Print statistics.
    print("Number of test frames = " + str(input_S.shape[0]))
    print("input_S shape: " + str(input_S.shape))  # Should be (test_size, 64, 64, 3).
    return input_S,input_C,X_test_orig_size


def main():
    # input_path = './exp2'
    # extractFrameOfVideo("./secret.mp4",frame_rate=5,frame_save_path=input_path+"/image/secret")
    # extractFrameOfVideo("./cover.mp4", frame_rate=5, frame_save_path=input_path + "/image/cover")

    input_S, input_C, X_test_orig_size = load_frames_preprocess(num_images_theta=30)
    batch_size = 2

    output_path = "./outcome_video"
    frame_num = input_S.shape[0]
    for i in range(frame_num//batch_size):
        print(i)
        if batch_size==1:
            input_S_small = input_S[i]
            input_C_small = input_C[i]
            name = [i]
        else:
            input_S_small = input_S[i*batch_size:i*batch_size+batch_size]
            input_C_small = input_C[i*batch_size:i*batch_size+batch_size]
            name = list(range(i*batch_size,i*batch_size+batch_size,1))

        validation(input_S_small, input_C_small, X_test_orig_size,save_path=output_path,name_box=name,disp=False)
    generateVideo(output_path+"/cover","cover_r.mp4",5)
    generateVideo(output_path + "/secret", "secret_r.mp4", 5)

if __name__ =="__main__":
    main()