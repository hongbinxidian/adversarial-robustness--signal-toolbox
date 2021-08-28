import numpy as np
import math
def snr_plot(model, snrs, lbl, test_idx, X_test, Y_test, classes):
    # Plot confusion matrix
    acc = {}
    for snr in snrs:

        # extract classes @ SNR
        test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        conf = np.zeros([len(classes), len(classes)])
        confnorm = np.zeros([len(classes), len(classes)])
        for i in range(0, test_X_i.shape[0]):
            j = list(test_Y_i[i, :]).index(1)
            k = int(np.argmax(test_Y_i_hat[i, :]))
            conf[j, k] = conf[j, k] + 1
        for i in range(0, len(classes)):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print(snr, "dB, Overall Accuracy: ", cor / (cor + ncor))
        acc[snr] = 1.0 * cor / (cor + ncor)
    return acc
def SNR_singlech(S, SN):
    S = S-np.mean(S)# 消除直流分量
    S = S/np.max(np.abs(S))#幅值归一化
    mean_S = (np.sum(S))/(len(S))#纯信号的平均值
    PS = np.sum((S-mean_S)*(S-mean_S))
    PN = np.sum((S-SN)*(S-SN))
    snr=10*math.log((PS/PN), 10)
    return(snr)