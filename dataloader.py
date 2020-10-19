import cv2,os
import numpy as np


class DataLoader():
    def loadData(self):
        data_path = 'D:\Downloads\observations-master\experiements\data'
        folders = os.listdir(data_path)
        labels = [i for i in range(len(folders))]
        label_dict = dict(zip(folders, labels))
        img_size = 100
        data = []
        label = []

        for category in folders:
            folder_path = os.path.join(data_path, category)
            img_names = os.listdir(folder_path)

            for img_name in img_names:
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (img_size, img_size))
                data.append(img)
                label.append(label_dict[category])



        data = np.array(data) / 255.0
        print(data.shape)
        data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
        label = np.array(label)
        print(data.shape)
        print(label.shape)
        return (data, label)
loader = DataLoader()
loader.loadData()