import os
import random
import numpy as np
from tqdm import tqdm


def find_class_folders(class_name):
    assert class_name in ["bird", "cat", "dog"]
    with open("data_setting/imagenet_hierarch.txt", 'r') as f:
        lines = f.readlines()
    class_folders = []
    for line in lines:
        command = line.split('\n')[0]
        command_splits = command.split(' ')
        folder_name = command_splits[2].split('/')[1]
        folder_class = command_splits[3].split('/')[1]
        if folder_class == class_name:
            class_folders.append(folder_name)
    print(class_folders)
    print(len(class_folders))
    return class_folders


def sample_images(root, class_folders, num_samples, save_path):
    num_subclass = len(class_folders)
    num_per_subclass = num_samples // num_subclass
    
    image_pairs = []
    for subclass in tqdm(range(num_subclass)):
        class_folder = class_folders[subclass]
        image_paths = os.listdir(os.path.join(root, class_folder))
        if subclass < num_samples % num_subclass:
            for image_path in image_paths[:num_per_subclass+1]:
                image_pairs.append(os.path.join(root, class_folder, image_path))
        else:
            for image_path in image_paths[:num_per_subclass]:
                image_pairs.append(os.path.join(root, class_folder, image_path))
    
    random.shuffle(image_pairs)

    with open(save_path, 'w') as f:
        for image_pair in image_pairs:
            f.write(image_pair)
            f.write('\n')
 
    return image_pairs
            


if __name__ == "__main__":
    class_folders = find_class_folders('dog')
    image_pairs = sample_images("datasets/imagenet/data/ILSVRC/Data/CLS-LOC/train", class_folders, num_samples=2000, save_path="datasets/imagenet/dog_2000.txt")
