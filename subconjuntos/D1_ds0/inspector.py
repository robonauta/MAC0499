import os
import sys
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label


def main():
    # Gets currect directory path
    cdir = os.getcwd()

    # Gets all files .png from the folder 'train'
    targets = glob.glob(str(cdir)+"/target/*.png")

    targets.sort()

    df = pd.DataFrame(columns=['image', '#non-text', '#text',
                               '#regions', 'average area', 'std area', 'average height', 'std height', 'average width', 'std width'])

    for i in range(0, len(targets)):
        img = plt.imread(targets[i])
        img = img.astype(np.uint8)

        labeled_image, regions_number = label(img, return_num=True)
        regions = regionprops(labeled_image)

        areas = []
        heights = []
        widths = []
        for props in regions:
            areas.append(props.area)
            heights.append(props.bbox[2] - props.bbox[0])
            widths.append(props.bbox[3] - props.bbox[1])

            '''
            print('Area: {0}'.format(areas[-1]))
            print('Height: {0}'.format(heights[-1]))
            print('Width: {0}'.format(widths[-1]))

            plt.imshow(props.image) 
            plt.show() 
            '''


        class_freq = np.unique(img, return_counts=True)[1]
        if len(class_freq) == 1:
            class_freq = np.append(class_freq, 0)

        print(targets[i][-19:])

        stats = pd.DataFrame({'image': [targets[i][-19:]], '#non-text': [class_freq[0]], '#text': [class_freq[1]], '#regions': [regions_number],  'average area': [np.average(areas)], 'std area': [
                             np.std(areas)],  'average height': [np.average(heights)], 'std height': [np.std(heights)],  'average width': [np.average(widths)], 'std width': [np.std(widths)]})
        df = df.append(stats)

    df.to_csv('img_stats.csv', index=False)


if __name__ == "__main__":
    main()
