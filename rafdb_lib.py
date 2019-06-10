import cv2, os, numpy as np, matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np
import tqdm
import re
from skimage.feature import hog, local_binary_pattern
from skimage import exposure

def plot_confusion_matrix(y_test, y_pred, classes=[], normalize=True, 
                          title='Average accuracy {acc:.2f}%\n', cmap=None, precision=2, text_size=10, title_size=25,
                          axis_label_size=16, tick_size=14, verbose = 1):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    plt.figure(figsize=(8,8))

    cm = confusion_matrix(y_test, y_pred)
    acc = sum(cm.diagonal() / (cm.sum() + 0.0001)) * 100.0
    if verbose == 1: print(cm)
    if normalize: cm = (cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis]+ 0.0001)) * 100.0
        
    plt.title(title.format_map({'acc': acc}), fontsize=title_size)
    sns.heatmap(cm, annot=True, fmt=".%df"%(precision), 
                cbar = False, square = True, cmap = cmap,
                xticklabels=classes, 
                yticklabels=classes)
    plt.ylabel('True label', fontsize=axis_label_size)
    plt.xlabel('Predicted label', fontsize=axis_label_size)
# plot_confusion_matrix

def lbp_feature_extraction(gray_image, num_points, radius, entire_image = True, verbose = 1):
    
    if entire_image == True:
        lbp = local_binary_pattern(gray_image, num_points, radius, method="uniform")
        
        # histogram & features
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 0.0001)
        
        if verbose == 1:
            print("LBP in entire image")
            print(lbp.shape, hist.shape)
            plt.subplot(1,2,1), plt.imshow(np.uint8(gray_image), cmap='gray'), plt.axis("off"), plt.title("Image")
            plt.subplot(1,2,2), plt.imshow(np.uint8(lbp), cmap='gray'), plt.axis("off"), plt.title("LBP")
            plt.show()
        # if
        
        return hist
    else:
        image_size = gray_image.shape[0:2]
        image_patch_height = int(gray_image.shape[0] / 10) # height = 10
        image_patch_width  = int(gray_image.shape[1] / 10) # width = 10
        
        image_patches = []
        for i in range(10):
            for j in range(10):
                # i * image_patch_height: (i + 1) * image_patch_height, j * image_patch_width: (j + 1) * image_patch_width
                # i: 0:10, 10:20, 20:30, 30:40, 40:50, 50:60, 60:70, 70:80, 80:90, 90:100
                # j: 0:10, 10:20, 20:30, 30:40, 40:50, 50:60, 60:70, 70:80, 80:90, 90:100
                image_patch = gray_image[i * image_patch_height: (i + 1) * image_patch_height, j * image_patch_width: (j + 1) * image_patch_width]
                image_patches.append(image_patch)
            # for
        # for

        patch_lbps   = []
        patch_hists  = []
        for patch_image in image_patches:
            patch_lbp = local_binary_pattern(patch_image, num_points, radius, method="uniform")

            (patch_hist, _) = np.histogram(patch_lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
            patch_hist  = patch_hist.astype("float")
            patch_hist /= (patch_hist.sum() + 0.0001)

            patch_lbps.append(patch_lbp)
            patch_hists.append(patch_hist)
        # for
        patch_total = np.hstack([patch for patch in patch_hists])
    
        if verbose == 1:
            print("LBP in patch images")
            print(patch_total.shape)

            plt.figure(figsize=(10, 10))
            for idx in range(len(image_patches)):
                plt.subplot(10, 10,idx + 1), plt.imshow(image_patches[idx], cmap='gray'), plt.axis("off")
            plt.show()

            plt.figure(figsize=(10, 10))
            for idx in range(len(image_patches)):
                plt.subplot(10, 10,idx + 1), plt.imshow(patch_lbps[idx], cmap='gray'), plt.axis("off")
            plt.show()
        # if
        
        return patch_total
# lbp_feature_extraction

def hog_feature_extraction(rgb_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), verbose = 1):
    feature, hog_image = hog(rgb_image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True, multichannel=True)

    if verbose == 1:
        print("Feature Shape: ", feature.shape)

        print("Feature Visualize: ")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(rgb_image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()
    # if
    
    return feature
# hog_dense_feature_extraction

def read_aligned_image(db, root_dir, image_id = "train_00010", verbose = 1):
    db_data = db.set_index(keys=["id"], drop=False)
    
    image_path = os.path.join(root_dir, db_data.loc[image_id]["image_aligned"]).replace("\\", "/")
    image = cv2.imread(image_path)
    image_info = db_data.loc[image_id].copy()
    
    if verbose == 1:
        plt.imshow(image[..., ::-1])

    return image, image_info
# read_aligned_image

def read_view_image(db, root_dir, image_id = "train_00010", get_only_face = True, 
                    draw_landmarks = True, draw_bbox = True, draw_points = True,
                    verbose = 1, fill_width = 5, line_width = 5):
    """
    Read image from dataset & annotation
    + read_view_image(df_rafdb, rafdb_info["root_dir"], get_only_face = True, draw_annotation = True, verbose = 1)
    """
    db_data = db.set_index(keys=["id"], drop=False)
    
    image_path = os.path.join(root_dir, db_data.loc[image_id]["image_original"]).replace("\\", "/")
    image = cv2.imread(image_path)
    image_info = db_data.loc[image_id].copy()
    
    image_info["bbox"][0] = np.clip(image_info["bbox"][0], 0, image.shape[1] - 1)
    image_info["bbox"][1] = np.clip(image_info["bbox"][1], 0, image.shape[0] - 1)
    image_info["bbox"][2] = np.clip(image_info["bbox"][2], image_info["bbox"][0], image.shape[1] - 1)
    image_info["bbox"][3] = np.clip(image_info["bbox"][3], image_info["bbox"][1], image.shape[0] - 1)
    image_info["bbox"] = image_info["bbox"].astype(dtype=np.int)
    
    
    if draw_landmarks == True:
        # Draw 37 landmarks
        for point in image_info["landmarks"]:
            cv2.circle(image, (int(point[0]), int(point[1])), fill_width, (255, 0, 0), -1)
            pass
        # for
    # if
        
    if draw_points == True:
        # Draw 5 basic alignment points
        cv2.circle(image, (int(image_info["eye1"][0]), int(image_info["eye1"][1])), fill_width, (0, 0, 255), -1)
        cv2.circle(image, (int(image_info["eye2"][0]), int(image_info["eye2"][1])), fill_width, (0, 0, 255), -1)
        cv2.circle(image, (int(image_info["nose"][0]), int(image_info["nose"][1])), fill_width, (0, 0, 255), -1)
        cv2.circle(image, (int(image_info["mouth1"][0]), int(image_info["mouth1"][1])), fill_width, (0, 0, 255), -1)
        cv2.circle(image, (int(image_info["mouth2"][0]), int(image_info["mouth2"][1])), fill_width, (0, 0, 255), -1)
    # if

    if draw_bbox == True:
        # Draw bounding box
        cv2.rectangle(image, (int(image_info["bbox"][0]), int(image_info["bbox"][1])), (int(image_info["bbox"][2]), int(image_info["bbox"][3])), (0, 0, 255), line_width)
    # if
    
    if get_only_face == True:
        image = image[int(image_info["bbox"][1]): int(image_info["bbox"][3]), int(image_info["bbox"][0]): int(image_info["bbox"][2])]
    # if
    
    if verbose == 1:
        plt.imshow(image[...,::-1])
    pass

    return image, image_info
# read_view_image

def build_rafdb_basic(rafdb_basic_dir = "./data/rafdb/basic", save_name = "rafdb_basic.hdf5", table_name = "data"):
    """
    INPUTS:
    + rafdb_basic_dir = "./data/rafdb/basic", 
    + save_name = "rafdb_basic.hdf5", 
    + table_name = "data"
    OUTPUTS:
    + ./data/rafdb/rafdb_basic.hdf5 with key = data
    EXAMPLE:
    + build_rafdb_basic()
    """
    
    # mapping dictionary from int --> string
    label_mapping  = {1: "Surprise", 2: "Fear", 3: "Disgust", 4: "Happiness", 5: "Sadness", 6: "Anger", 7: "Neutral"}
    gender_mapping = {0: "male", 1: "female", 2: "unsure"}
    race_mapping   = {0: "Caucasian", 1: "African-American", 2: "Asian"}
    age_mapping    = {0: "0-3", 1: "4-19", 2: "20-39", 3: "40-69", 4: "70+"}

    # data table structure
    db_rafdb_basic_data   = {"id": [], "type": [], "emotion": [], "image_aligned": [], "image_original": [], "bbox": [], 
                                         "eye1": [], "eye2": [], "nose": [], "mouth1": [], "mouth2": [], "gender": [], "race": [], "age": [], "landmarks": []}
    db_rafdb_basic_column = ["id", "type", "emotion", "image_aligned", "image_original", "bbox",  
                             "eye1", "eye2", "nose", "mouth1", "mouth2", "gender", "race", "age", "landmarks"]

    print("Step1. Read train emotion data")
    db_emo_label = pd.read_csv(os.path.join(rafdb_basic_dir, "EmoLabel/list_patition_label.txt"), header=None, delimiter=" ", 
                               names = ["file", "emo"], usecols=["file", "emo"])
    db_emo_label = db_emo_label.set_index(keys=["file"], drop=False)

    for i in tqdm.tqdm(range(1, 12272), desc = "Read train"):
        v_idx    = "train_%05d"%(i)
        v_type   = "train"
        
        ## path of original & aligned image ##
        v_image_align    = "Image/aligned/%s_aligned.jpg"%(v_idx)
        v_image_original = "Image/original/%s.jpg"%(v_idx)

        ## bbox ##
        v_anno_bbox      = "Annotation/boundingbox/%s_boundingbox.txt"%(v_idx)
        v_bbox           = pd.read_csv(os.path.join(rafdb_basic_dir, v_anno_bbox), header=None, delimiter=" ").values[0, 0 : 4]

        ## emo ##
        v_emo            = db_emo_label.loc['%s.jpg'%(v_idx)]["emo"]

        ## manu_attri ##
        v_attri_path = "Annotation/manual/%s_manu_attri.txt"%(v_idx)
        with open(os.path.join(rafdb_basic_dir, v_attri_path), "rt") as attri_file: v_attri_content = attri_file.readlines()
        v_eye1_pos   = np.array([float(x) for x in re.split('\t| |;|,|\n', v_attri_content[0])[0:2]])
        v_eye2_pos   = np.array([float(x) for x in re.split('\t| |;|,|\n', v_attri_content[1])[0:2]])
        v_nose_pos   = np.array([float(x) for x in re.split('\t| |;|,|\n', v_attri_content[2])[0:2]])
        v_mouth1_pos = np.array([float(x) for x in re.split('\t| |;|,|\n', v_attri_content[3])[0:2]])
        v_mouth2_pos = np.array([float(x) for x in re.split('\t| |;|,|\n', v_attri_content[4])[0:2]])
        v_gender     = int(v_attri_content[5])
        v_race       = int(v_attri_content[6])
        v_age        = int(v_attri_content[7])

        ## auto landmarks from Face++ API ##
        v_auto_path      = "Annotation/auto/%s_auto_attri.txt"%(v_idx)
        v_auto_landmarks = pd.read_csv(os.path.join(rafdb_basic_dir, v_auto_path), header=None, delimiter="\t").values

        ## append ##
        db_rafdb_basic_data["id"].append(v_idx)
        db_rafdb_basic_data["type"].append(v_type)
        db_rafdb_basic_data["image_aligned"].append(v_image_align)
        db_rafdb_basic_data["image_original"].append(v_image_original)
        db_rafdb_basic_data["bbox"].append(v_bbox)
        db_rafdb_basic_data["emotion"].append(v_emo)

        db_rafdb_basic_data["eye1"].append(v_eye1_pos)
        db_rafdb_basic_data["eye2"].append(v_eye2_pos)
        db_rafdb_basic_data["nose"].append(v_nose_pos)
        db_rafdb_basic_data["mouth1"].append(v_mouth1_pos)
        db_rafdb_basic_data["mouth2"].append(v_mouth2_pos)
        db_rafdb_basic_data["gender"].append(v_gender)
        db_rafdb_basic_data["race"].append(v_race)
        db_rafdb_basic_data["age"].append(v_age)

        db_rafdb_basic_data["landmarks"].append(v_auto_landmarks)
    # for

    print("Step2. Read test emotion data")
    for i in tqdm.tqdm(range(1, 3069), desc = "Read test"):
        v_idx    = "test_%04d"%(i)
        v_type   = "test"
        
        ## path of original & aligned image ##
        v_image_align    = "Image/aligned/%s_aligned.jpg"%(v_idx)
        v_image_original = "Image/original/%s.jpg"%(v_idx)

        ## bbox ##
        v_anno_bbox      = "Annotation/boundingbox/%s_boundingbox.txt"%(v_idx)
        v_bbox           = pd.read_csv(os.path.join(rafdb_basic_dir, v_anno_bbox), header=None, delimiter=" ").values[0, 0 : 4]

        ## emo ##
        v_emo            = db_emo_label.loc['%s.jpg'%(v_idx)]["emo"]

        ## manu_attri ##
        v_attri_path = "Annotation/manual/%s_manu_attri.txt"%(v_idx)
        with open(os.path.join(rafdb_basic_dir, v_attri_path), "rt") as attri_file: v_attri_content = attri_file.readlines()
        v_eye1_pos   = np.array([float(x) for x in re.split('\t| |;|,|\n', v_attri_content[0])[0:2]])
        v_eye2_pos   = np.array([float(x) for x in re.split('\t| |;|,|\n', v_attri_content[1])[0:2]])
        v_nose_pos   = np.array([float(x) for x in re.split('\t| |;|,|\n', v_attri_content[2])[0:2]])
        v_mouth1_pos = np.array([float(x) for x in re.split('\t| |;|,|\n', v_attri_content[3])[0:2]])
        v_mouth2_pos = np.array([float(x) for x in re.split('\t| |;|,|\n', v_attri_content[4])[0:2]])
        v_gender     = int(v_attri_content[5])
        v_race       = int(v_attri_content[6])
        v_age        = int(v_attri_content[7])

        ## auto landmarks from Face++ API ##
        v_auto_path      = "Annotation/auto/%s_auto_attri.txt"%(v_idx)
        v_auto_landmarks = pd.read_csv(os.path.join(rafdb_basic_dir, v_auto_path), header=None, delimiter="\t").values

        ## append ##
        db_rafdb_basic_data["id"].append(v_idx)
        db_rafdb_basic_data["type"].append(v_type)
        db_rafdb_basic_data["image_aligned"].append(v_image_align)
        db_rafdb_basic_data["image_original"].append(v_image_original)
        db_rafdb_basic_data["bbox"].append(v_bbox)
        db_rafdb_basic_data["emotion"].append(v_emo)

        db_rafdb_basic_data["eye1"].append(v_eye1_pos)
        db_rafdb_basic_data["eye2"].append(v_eye2_pos)
        db_rafdb_basic_data["nose"].append(v_nose_pos)
        db_rafdb_basic_data["mouth1"].append(v_mouth1_pos)
        db_rafdb_basic_data["mouth2"].append(v_mouth2_pos)
        db_rafdb_basic_data["gender"].append(v_gender)
        db_rafdb_basic_data["race"].append(v_race)
        db_rafdb_basic_data["age"].append(v_age)

        db_rafdb_basic_data["landmarks"].append(v_auto_landmarks)
    # for

    print("Step3. Build RAF-DB Basic")
    db = pd.DataFrame(db_rafdb_basic_data, columns=db_rafdb_basic_column)
    db.to_hdf(os.path.join(rafdb_basic_dir, save_name), table_name)
    
    return db
# build_rafdb_basic