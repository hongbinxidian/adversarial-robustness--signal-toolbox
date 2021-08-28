"""
The script demonstrates a simple example of using ART with Keras. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Activation, Dropout
import numpy as np
from numpy import *
from models.conv_resnet import conv_resnet
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, DeepFool, ShadowAttack, SaliencyMapMethod, UniversalPerturbation
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist
import matplotlib.pyplot as pl
import os
from data.rmldataset2016 import load_data, load_data_snrs
import argparse
import pickle
import logging
import math
from snr import snr_plot
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-m', '--attack_model', default='UniversalPerturbation',
                    help='FastGradientMethod ,BasicIterativeMethod, SaliencyMapMethod, UniversalPerturbation version.')
args = parser.parse_args()

def SNR_singlech(S, SN):
    S = S-np.mean(S)# 消除直流分量
    S = S/np.max(np.abs(S))#幅值归一化
    mean_S = (np.sum(S))/(len(S))#纯信号的平均值
    PS = np.sum((S-mean_S)*(S-mean_S))
    PN = np.sum((S-SN)*(S-SN))
    snr=10*math.log((PS/PN), 10)
    return(snr)



def plot_signal(signal, signal1, path):

    pl.plot([i for i in range(0, 128, 1)], signal)
    pl.title("original images")
    image_name = os.path.join(path, "original images.jpg")
    pl.savefig(image_name)
    pl.close('all')

    pl.plot([i for i in range(0, 128, 1)], signal1)
    pl.title("attacked images")
    image_name = os.path.join(path, "attacked images.jpg")
    pl.savefig(image_name)
    pl.close('all')


# Step 1: Load the MNIST dataset
snr_define = 0
(mods, snrs, lbl), (X_train, Y_train), (X_test, Y_test), (train_idx, test_idx), (min_pixel_value, max_pixel_value) = load_data_snrs(
        '/media/xd2/Signal/mhb/xdaima-adversarial-robustness-toolbox-dev_1.5.0/adversarial-robustness-toolbox/data/RML2016.10a_dict.pkl', [i for i in range(snr_define, 20, 2)], train_rate=0.9)
# (mods, snrs, lbl), (X_train, Y_train), (X_test, Y_test), (train_idx, test_idx), (min_pixel_value, max_pixel_value) = load_data(
#         '/media/xd2/Signal/mhb/xdaima-AdvBox-master/AdvBox/data/RML2016.10a_dict.pkl', train_rate=0.9)
input_shape=(128, 2)
# Step 2: Create the model
dr = 0.5
classes = len(mods)
model = conv_resnet(mods, weights="/media/xd2/Signal/mhb/xdaima-adversarial-robustness-toolbox-dev_1.5.0/adversarial-robustness-toolbox/models/conv_resnet.whs.h5", input_shape = (128, 2), classes=11)


# Step 3: Create the ART classifier

classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)

# Step 4: Train the ART classifier

# classifier.fit(X_train, Y_train, batch_size=256, nb_epochs=5)

path = "/media/xd2/Signal/mhb/xdaima-adversarial-robustness-toolbox-dev_1.5.0/adversarial-robustness-toolbox/data"
path1 = os.path.join(path, args.attack_model)
if not os.path.exists(path1):
    os.mkdir(path1)

# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)
# handler = logging.FileHandler(path1+"/log.txt")
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
#
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# #
# logger.addHandler(handler)
# logger.addHandler(console)

# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(X_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(Y_test, axis=1)) / len(Y_test)
# logger.info("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
if args.attack_model == "FastGradientMethod":
    attack = FastGradientMethod(estimator=classifier, eps=0.001)
elif args.attack_model == "BasicIterativeMethod":
    attack = BasicIterativeMethod(estimator=classifier, eps=0.01)
elif args.attack_model == "SaliencyMapMethod":
    attack = SaliencyMapMethod(classifier, theta = 0.01,gamma = 0.01, batch_size=256)
elif args.attack_model == "UniversalPerturbation":
    attack = UniversalPerturbation(classifier=classifier, attacker="fgsm", batch_size=256, eps=0.003, delta=0.7)

# x_test_adv = attack.generate(x=X_test)
#
# # Step 7: Evaluate the ART classifier on adversarial test examples
#
# predictions = classifier.predict(x_test_adv)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(Y_test, axis=1)) / len(Y_test)
# logger.info("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
#
# data = {"original signal":X_test, "attack_signals":x_test_adv, "true_labels":Y_test, }
# f = open(path1+"/data.pkl", mode = 'wb')
# pickle.dump(data, f)
# f.close()

# snrs1 = []
# for i in range(x_test_adv.shape[0]):
    # path2 = os.path.join(path1, str(i))
    # if not os.path.exists(path2):
    #     os.mkdir(path2)
    # plot_signal(X_test[i], x_test_adv[i], path2)
    # snrs1.append(SNR_singlech(X_test[i], x_test_adv[i]))
# logger.info("SNR for all signals: {}%".format(mean(snrs1)))
# logger.info(snr_plot(model, snrs, lbl, test_idx, x_test_adv, Y_test, mods))
print(snr_plot(model, snrs, lbl, test_idx, X_test, Y_test, mods))


