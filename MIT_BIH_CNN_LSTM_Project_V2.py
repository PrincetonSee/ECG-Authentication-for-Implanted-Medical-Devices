from glob import glob
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd
import tensorflow as tf
import biosppy
import os
import cv2
import matplotlib.pyplot as plt
import wfdb
import random

numfolds = 10
seed = 42
np.random.seed(seed)


def get_records():
    """ Get paths for data in data/mit/ directory """
    # Download if doesn't exist
    # There are 3 files for each record
    # *.atr is one of them
    paths = glob('data/*.atr')

    # Get rid of the extension
    paths = [path[:-4] for path in paths]
    random.shuffle(paths)
    return paths[:10]


def segmentation(records):
    # Normal = []
    # Abnormal = []

    mainPatient = []
    otherPatients = []

    isMainPatient = True

    for e in records:
        signals, fields = wfdb.rdsamp(e, channels=[0])
        ann = wfdb.rdann(e, 'atr')
        all_beats = ann.sample[:]
        beats = ann.sample

        if isMainPatient:
            for i in all_beats:
                beats = list(beats)
                j = beats.index(i)
                if j != 0 and j != (len(beats) - 1):
                    # print(j-1)
                    x = beats[j - 1]
                    y = beats[j + 1]
                    diff1 = abs(x - beats[j]) // 2
                    diff2 = abs(y - beats[j]) // 2
                    a = signals[beats[j] - diff1: beats[j] + diff2, 0]
                    for k in a:
                        mainPatient.append(k)
            isMainPatient = False
        else:
            for i in all_beats:
                beats = list(beats)
                j = beats.index(i)
                if j != 0 and j != (len(beats) - 1):
                    # print(j-1)
                    x = beats[j - 1]
                    y = beats[j + 1]
                    diff1 = abs(x - beats[j]) // 2
                    diff2 = abs(y - beats[j]) // 2
                    a = signals[beats[j] - diff1: beats[j] + diff2, 0]
                    for k in a:
                        otherPatients.append(k)
    return np.array(mainPatient, dtype=np.float), np.array(otherPatients, dtype=np.float)


def save_to_csv(beats, dir):
    # Save to CSV file.
    # print(list(beats[:]))
    # savedata = np.array(list(beats[:]), dtype=np.float)
    outfn = dir
    print('    Generating ', outfn)
    with open(outfn, "wb") as fin:
        np.savetxt(fin, beats, delimiter=",", fmt='%f')
        print('        saved file ', outfn)


def create_signal(file_name):
    csv = pd.read_csv(file_name, names=["values"])
    # df = DataFrame(csv, columns=['Test'])
    # print(csv.shape)
    # print(len(csv))
    # print(csv)
    # print(csv[' Sample Value'])
    csv_data = csv['values']
    data = np.array(csv_data)
    # print(data)
    # print(data.shape)
    # print(len(data))
    signals = []
    count = 1
    peaks = biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate=200)[0]
    for i in (peaks[1:-1]):
        diff1 = abs(peaks[count - 1] - i)
        diff2 = abs(peaks[count + 1] - i)
        x = peaks[count - 1] + diff1 // 2
        y = peaks[count + 1] - diff2 // 2
        signal = data[x:y]
        signals.append(signal)
        count += 1
    return signals


def signal_to_img(array, directory):
    if os.path.exists(directory):
        print('This directory exists, overwriting directory.')
    else:
        os.makedirs(directory)

    for count, i in enumerate(array):
        fig = plt.figure(frameon=False)
        plt.plot(i)
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        filename = directory + '/' + str(count) + '.png'
        fig.savefig(filename)
        im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(filename, im_gray)
        plt.close()
    return directory


def generate_input_files(mainData, otherData):
    nData = glob(mainData + '/*.png')
    aData = glob(otherData + '/*.png')

    n_formatted = []
    o_formatted = []
    for n in nData:
        n_formatted.append(n + " 1")
    for a in aData:
        o_formatted.append(a + " 0")

    random.shuffle(n_formatted)
    random.shuffle(o_formatted)
    first_half_n_formatted = n_formatted[:1000]
    some_o_formatted = o_formatted[:3724]
    testFiles = first_half_n_formatted + some_o_formatted
    trainFiles = n_formatted[1000:] + o_formatted[3724:]
    # random.shuffle(total_files)
    # # print(len(some_o_formatted))
    # rest_o_formatted = o_formatted[1000:]
    # # print(len(rest_o_formatted))
    # testFiles = first_half_n_formatted + some_o_formatted
    # random.shuffle(testFiles)
    # trainFiles = second_half_n_formatted + rest_o_formatted
    # random.shuffle(trainFiles)
    # # print(len(trainFiles))

    with open('training_files.txt', 'w') as train:
        for t in trainFiles:
            train.write(t + '\n')

    with open('testing_files.txt', 'w') as test:
        for t in testFiles:
            test.write(t + '\n')

    # with open('total_files.txt', 'w') as total:
    #     for t in total_files:
    #         total.write(t + '\n')


def create_data_label(filename):
    file = open(filename, 'r')
    lines = file.readlines()

    x_train = np.zeros((len(lines), 4, 128, 128, 1))
    x_label = np.zeros(len(lines), dtype='int')

    for i in range(len(lines)):
        print("loading image: " + str(i))
        path = lines[i].split(" ")[0]
        label = lines[i].split(" ")[-1]

        label = label.strip('\n')
        label = int(label)
        x_label[i] = label

        img_raw = tf.io.read_file(path)
        img_tensor = tf.image.decode_image(img_raw)
        img_final = tf.image.resize(img_tensor, [128, 128])
        # print(img_final)
        # print([128] + img_tensor)
        # print(img_tensor.shape)
        x_train[i] = [128] + img_final
        # print(x_train[0].dtype)
        # print(x_train[1].dtype)
        # print(x_train[2].dtype)
        # print(x_train[3].dtype)
        # print(x_train[4].dtype)
    # print(len(x_train))
    return x_train, x_label


def build_cnn_model(height, width):
    Conv2D = tf.keras.layers.Conv2D
    MaxPool2D = tf.keras.layers.MaxPool2D
    Flatten = tf.keras.layers.Flatten
    Dropout = tf.keras.layers.Dropout
    Dense = tf.keras.layers.Dense
    TimeDistributed = tf.keras.layers.TimeDistributed
    LSTM = tf.keras.layers.LSTM
    BatchNormalization = tf.keras.layers.BatchNormalization
    Activation = tf.keras.layers.Activation

    model = tf.keras.Sequential()

    # model.add(Conv2D(32, (3, 3), input_shape=(height, width, 1)))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 1)))
    model.add(MaxPool2D(2, 2))
    # model.add(Conv2D(64, (3, 3)))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))
    # model.add(Conv2D(64, (3, 3)))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))

    model.add(Flatten())
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(1, activation='softmax'))

    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model


def build_lstm_model(cnn):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.TimeDistributed(cnn, input_shape=(4, 128, 128, 1)))
    model.add(tf.keras.layers.LSTM(4))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model


if __name__ == '__main__':
    # get MIT_BIH records
    print("getting records")
    paths = get_records()
    # identify normal and abnormal beats
    print("segmenting beats")
    mainPatient, otherPatients = segmentation(paths)
    # store beats in csv file
    print("saving to csv")
    save_to_csv(mainPatient, 'mit_bih_main_patient.csv')
    save_to_csv(otherPatients, 'mit_bih_other_patients.csv')
    # convert files to ecg signals
    print("converting to signals")
    normal_signals = create_signal('mit_bih_main_patient.csv')
    abnormal_signals = create_signal('mit_bih_other_patients.csv')
    # create ecg images from signals
    print("creating images")
    normal_directory = signal_to_img(normal_signals, "ECG_Images_Main_Patient")
    abnormal_directory = signal_to_img(abnormal_signals, "ECG_Images_Other_Patients")
    # create train/val/test txt files form ecg images
    print('generating input text files')
    generate_input_files("ECG_Images_Main_Patient", "ECG_Images_Other_Patients")
    print("generating training data/labels")
    x_train, y_train = create_data_label('training_files.txt')
    x_test, y_test = create_data_label('testing_files.txt')
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    cvscores = []
    cnn_model = build_cnn_model(128, 128)
    cnn_lstm_model = build_lstm_model(cnn_model)
    counter = 0
    with open('confusionMatrixAdam.txt', 'w') as matrix:
        for train, test in kfold.split(x_train, y_train):
            print("Training Model")
            cnn_lstm_model.fit(x_train[train], y_train[train], batch_size=32, epochs=1)

            predictions = cnn_lstm_model.predict_classes(x_test, batch_size=32)
            true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_test, predictions).ravel()
            print("true_neg = " + str(true_neg))
            print("true_pos = " + str(true_pos))
            print("false_pos = " + str(false_pos))
            print("false_neg = " + str(false_neg))

            scores = cnn_lstm_model.evaluate(x_test, y_test, verbose=0, batch_size=32)
            print("%s: %.2f%%" % (cnn_lstm_model.metrics_names[1], scores[1] * 100))
            cvscores.append(scores[1] * 100)

            matrix.write("Adam " + str(counter) + '\n')
            matrix.write("true_neg = " + str(true_neg) + '\n')
            matrix.write("true_pos = " + str(true_pos) + '\n')
            matrix.write("false_pos = " + str(false_pos) + '\n')
            matrix.write("false_neg = " + str(false_neg) + '\n')
            matrix.write("%s: %.2f%%" % (cnn_lstm_model.metrics_names[1], scores[1] * 100))
            matrix.write("\n")
            counter += 1

    print(cvscores)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    test_loss, test_acc = cnn_lstm_model.evaluate(x_test, y_test)
    print(test_acc)
    print(test_loss)
