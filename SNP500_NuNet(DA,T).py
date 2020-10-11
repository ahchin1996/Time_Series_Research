import os
import time
import random
from datetime import date
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import choices
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv1D, concatenate, Reshape, LSTM, \
    Permute, Input, MaxPooling2D, Flatten, Dense, Activation, ConvLSTM2D
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

direction = "./SNP500_NuNet_DA_T/SNP500_NuNet_"
if not os.path.isdir("SNP500_NuNet_DA_T"):
    os.mkdir("SNP500_NuNet_DA_T")
seq_length = 20
mini_batch_size = 20
initial_epoch = 350
initial_lr = 0.002
initial_decay = 2e-9
online_epoch = 100
online_lr = 0.0005
online_decay = 3e-8
thresh_hold = 0.15
transaction_cost = 0.0003  # 0.3%

total_time_step = 4000
train_ratio = 0.7
validation_ratio = 0.1
test_ratio = 0.2
online_test_partition = 8
unit_length = total_time_step * test_ratio / online_test_partition
unit_iter = int(unit_length / mini_batch_size)
train_min = int(total_time_step * train_ratio / unit_length)
train_max = int(total_time_step * (1 - validation_ratio) / unit_length)

MSE = []
MAPE = []
MAE = []
BH_profit = []
Model_profit = []
Arg_min = []
Arg_min_test = []
overall_prediction = []
overall_target = []


def construct_model():
    input1 = Input(shape=(seq_length, 5790, 6))
    input2 = Input(shape=(seq_length, 6))

    re_input1 = Reshape((5790 * seq_length, 6))(input1)

    rere_input1 = Permute((2, 1), input_shape=(5790 * seq_length, 6))(re_input1)

    conv1 = Conv1D(30, 1, strides=1, padding='valid', activation='relu', data_format="channels_first", name='X1_input')(
        rere_input1)
    conv2 = Conv1D(30, 1, strides=1, padding='valid', activation='relu', data_format="channels_first", name='Conv7')(
        conv1)

    LSTM1 = LSTM(4, return_sequences=True)(input2)
    LSTM2 = LSTM(4, return_sequences=False)(LSTM1)

    reshape_conv2 = Reshape((30, 5790, seq_length))(conv2)

    pool = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid', data_format="channels_last")(reshape_conv2)

    reshape1 = Reshape((1, 30, 2895, seq_length))(pool)
    reshape2 = Permute((4, 2, 3, 1), input_shape=(1, 30, 2895, seq_length))(reshape1)

    convLSTM1 = ConvLSTM2D(filters=10, kernel_size=(3, 3), strides=(3, 3),
                           padding='same', return_sequences=True)(reshape2)
    convLSTM2 = ConvLSTM2D(filters=20, kernel_size=(3, 2), strides=(2, 2),
                           padding='same', return_sequences=True)(convLSTM1)
    convLSTM3 = ConvLSTM2D(filters=40, kernel_size=(3, 1), strides=(2, 2),
                           padding='same', return_sequences=True)(convLSTM2)
    convLSTM4 = ConvLSTM2D(filters=40, kernel_size=(2, 2), strides=(2, 2),
                           padding='same', return_sequences=False)(convLSTM3)

    flat1 = Flatten()(convLSTM4)
    flat2 = Flatten()(LSTM2)

    dense1 = Dense(120)(flat1)
    activation1 = Activation('relu')(dense1)
    merge2 = concatenate([activation1, flat2])
    dense2 = Dense(30)(merge2)
    activation2 = Activation('relu')(dense2)
    output = Dense(1, kernel_regularizer=regularizers.l2(0.000001))(activation2)

    model = Model(inputs=[input1, input2], outputs=[output])
    sgd = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    model = multi_gpu_model(model, gpus=2)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    print(model.summary())
    return model


def random_sample(a):
    np.random.shuffle(a)
    return a


def shuffle_batch(pop, weight):
    rnd_batch = choices(pop, weight)
    return rnd_batch


def load_and_out(a, kk, Seq_length, Mini_batch, SNP500_input, SNP500_target):
    Input_Companys = np.zeros(shape=[Mini_batch + Seq_length - 1, 5790, 6])
    for i in range(5790):
        Input_Companys[:, i, :] = a[i][Mini_batch * kk:(kk + 1) * Mini_batch + Seq_length - 1, 1:]
    dataX1 = []
    dataX2 = []
    dataY = np.zeros(shape=[Mini_batch])
    dataY[:] = SNP500_target[Mini_batch * kk:(kk + 1) * Mini_batch, 1]
    for k in range(0, Mini_batch):  # This loop is for getting the training set
        _x = SNP500_input[kk * Mini_batch + k: kk * Mini_batch + k + Seq_length,
             1:7]  # Save by window size(length of Sequence) to _x
        dataX2.append(_x)
    for k in range(0, len(Input_Companys) + 1 - Seq_length):  # This loop is for getting the training and validation set
        _x1 = Input_Companys[k:k + Seq_length, :, 0:6]
        dataX1.append(_x1)
    return dataX1, dataX2, dataY


# Back Testing
def back_test(thresh_hold, transaction_cost, target, prediction):
    cum_profit_Model = 0  # 0%
    cum_profit_BH = 0  # 0%
    profit_model = []
    profit_BH = []
    for i in range(len(target) - 1):
        cum_profit_BH += (target[i + 1] - target[i]) / target[i] * 100 - transaction_cost
        profit_BH.append(cum_profit_BH)
        if (prediction[i + 1] - target[i]) / target[i] * 100 > thresh_hold:  # Buy strategy
            cum_profit_Model += (target[i + 1] - target[i]) / target[i] * 100 - transaction_cost
            profit_model.append(cum_profit_Model)
        elif (prediction[i + 1] - target[i]) / target[i] * 100 < -thresh_hold:  # Short sell strategy
            cum_profit_Model += -(target[i + 1] - target[i]) / target[i] * 100 - transaction_cost
            profit_model.append(cum_profit_Model)
        else:  # Hold
            cum_profit_Model += 0
            profit_model.append(cum_profit_Model)
    plt.plot(profit_BH, label='Buy and Hold')
    plt.plot(profit_model, label='Model')
    plt.legend()
    plt.grid(color='g', linestyle='-', linewidth=1)
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Profit")
    plt.show()
    print("Back trading test results, Buy and Hold: {0:3f}, Model: {1:3f}" \
          .format(cum_profit_BH, cum_profit_Model))
    return cum_profit_BH, cum_profit_Model


def test_model(model, SNP500_input, SNP500_target, All_chart_info, \
               Max_close, Min_close, Mini_batch_size, start, end):
    TestLoss_iter = np.zeros(shape=[(end - start) * Mini_batch_size])
    prediction = np.zeros(shape=[(end - start) * Mini_batch_size, 1])
    target = np.zeros(shape=[(end - start) * Mini_batch_size])
    ii = 0
    for i in range(start, end):
        dataX1, dataX1, dataY = \
            load_and_out(All_chart_info, i, 20, Mini_batch_size, SNP500_input, SNP500_target)
        dataX = [dataX1, dataX2]
        TestLoss_iter[ii] = model.test_on_batch(x=dataX, y=dataY)
        prediction[(Mini_batch_size * ii):(Mini_batch_size * (ii + 1)), :] = model.predict_on_batch(dataX)
        target[Mini_batch_size * ii:Mini_batch_size * (ii + 1)] = dataY
        ii = ii + 1
    prediction = np.reshape(prediction, len(prediction))
    prediction = prediction * (Max_close - Min_close) + Min_close
    target = target * (Max_close - Min_close) + Min_close
    error = abs(prediction - target)
    MSE = pow(error, 2).mean()
    MAE = error.mean()
    MAPE = (error / target * 100).mean()
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(np.linspace(0, (end - start) * Mini_batch_size - 1, (end - start) * Mini_batch_size) \
             , prediction[:(end - start) * Mini_batch_size], linewidth=1, label='Prediction')
    plt.plot(np.linspace(0, (end - start) * Mini_batch_size - 1, (end - start) * Mini_batch_size) \
             , target[:(end - start) * Mini_batch_size], linewidth=1, label='Ground Truth')
    plt.legend()
    plt.grid(color='g', linestyle='-', linewidth=1)
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.show()
    print("MSE: {0:6f}, MAPE: {1:6f}, MAE: {2:6f}".format(MSE, MAPE, MAE))
    return MSE, MAPE, MAE, target, prediction


def test_model_final(prediction, target, Mini_batch_size, start, end):
    error = abs(prediction - target)
    MSE = pow(error, 2).mean()
    MAE = error.mean()
    MAPE = (error / target * 100).mean()
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(np.linspace(0, (end - start) * Mini_batch_size - 1, (end - start) * Mini_batch_size) \
             , prediction[:(end - start) * Mini_batch_size], linewidth=1, label='Prediction')
    plt.plot(np.linspace(0, (end - start) * Mini_batch_size - 1, (end - start) * Mini_batch_size) \
             , target[:(end - start) * Mini_batch_size], linewidth=1, label='Ground Truth')
    plt.legend()
    plt.grid(color='g', linestyle='-', linewidth=1)
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.show()
    print("MSE: {0:6f}, MAPE: {1:6f}, MAE: {2:6f}".format(MSE, MAPE, MAE))
    return MSE, MAPE, MAE


model = construct_model()
for k in range(train_min, train_max):
    # Get all chart information
    All_chart_info = []
    for i in range(1, 7):
        with open("./Chart_after_processing_US/All_Chart_info_US{}_window{}".format(i, k) + ".txt",
                  "rb") as fp:  # Unpickle preprocessed data
            b = pickle.load(fp)
        All_chart_info = All_chart_info + b
    SNP500_input = np.loadtxt("./Chart_after_processing_US/SNP500_input_window{}.csv".format(k), delimiter=',')
    SNP500_target = np.loadtxt("./Chart_after_processing_US/SNP500_target_window{}.csv".format(k), delimiter=',')
    if k == train_min:
        EPOCH = initial_epoch
        current_learning_rate = initial_lr
        decay = initial_decay
    else:
        EPOCH = online_epoch
        current_learning_rate = online_lr
        decay = online_decay
    TrainingLoss_epoch = np.zeros(shape=[EPOCH])
    ValidationLoss_epoch = np.zeros(shape=[EPOCH])
    Learning_rate = np.zeros(shape=[EPOCH])
    Max = unit_iter * k
    pop = np.linspace(0, Max - 1, Max)
    aMax = 2 / Max - 1 / Max * 1 / 100
    weight = np.linspace(1 / Max * 1 / 100, aMax, Max)
    iterations = 0
    for epoch in range(EPOCH):
        print("PARENT EPOCH: {}".format(epoch))
        TrainingLoss_iter = []
        ValidationLoss_iter = []

        for i in range(unit_iter * k):
            if epoch >= 50 and k == train_min:
                current_learning_rate = current_learning_rate / (1 + decay * iterations)
                K.set_value(model.optimizer.lr, current_learning_rate)
            elif k > train_min:
                current_learning_rate = current_learning_rate / (1 + decay * iterations)
                K.set_value(model.optimizer.lr, current_learning_rate)
            Shuffled = random_sample(All_chart_info)  # shuffle training data set
            # load new training set
            rnd_batch = int(shuffle_batch(pop, weight)[0])
            dataX1, dataX2, dataY \
                = load_and_out(Shuffled, rnd_batch, seq_length, mini_batch_size, SNP500_input, SNP500_target)
            dataX = [dataX1, dataX2]
            TrainingLoss = model.train_on_batch(x=dataX, y=dataY)
            TrainingLoss_iter.append(TrainingLoss)
            print("Window: {0:2d}, Epoch: {1:3d}, MiniBatch#: {2:4d}, TrainingLoss: {3:.10f}, LearningRate: {4: .6f}" \
                  .format(k, epoch, rnd_batch, TrainingLoss, K.eval(model.optimizer.lr)))
            iterations += 1

        for i in range(unit_iter * k, unit_iter * k + unit_iter * 4):
            dataX1, dataX2, dataY \
                = load_and_out(Shuffled, i, seq_length, mini_batch_size, SNP500_input, SNP500_target)
            dataX = [dataX1, dataX2]
            ValidationLoss = model.test_on_batch(x=dataX, y=dataY)
            ValidationLoss_iter.append(ValidationLoss)
            print("Window: {0:2d}, Epoch: {1:4d}, Iteration: {2:4d}, ValidationLoss: {3:.10f}" \
                  .format(k, epoch, i, ValidationLoss))

        TrainingLoss_epoch[epoch] = (np.array(TrainingLoss_iter)).mean()
        ValidationLoss_epoch[epoch] = (np.array(ValidationLoss_iter)).mean()
        TotalLoss_epoch = ValidationLoss_epoch + TrainingLoss_epoch
        Learning_rate[epoch] = K.eval(model.optimizer.lr)
        today = date.today()
        now = time.gmtime(time.time())
        model.save(direction + "Window:_{}_Epoch:{}.h5".format(k, epoch))
        np.savetxt(direction + "Window:_{}_LearningRate.csv".format(k), Learning_rate, delimiter=',')
        np.savetxt(direction + "Window:_{}_TrainingLoss.csv".format(k), TrainingLoss_epoch, delimiter=',')
        np.savetxt(direction + "Window:_{}_ValidationLoss.csv".format(k), ValidationLoss_epoch, delimiter=',')
        print("Window: {0:2d}, Epoch: {1:4d}, TrainingLoss: {2:.10f}, ValidationLoss: {3:.10f}" \
              .format(k, epoch, TrainingLoss_epoch[epoch], ValidationLoss_epoch[epoch]))
        if ((epoch + 1) % 2 == 0):
            plt.figure(1)
            plt.plot(np.linspace(0, epoch, epoch + 1), TrainingLoss_epoch[0:epoch + 1], \
                     'o-', label='TrainingLoss', markersize=4)
            plt.plot(np.linspace(0, epoch, epoch + 1), ValidationLoss_epoch[0:epoch + 1], \
                     'ro-', label='ValidationLoss', markersize=4)
            plt.legend()
            plt.grid(color='g', linestyle='-', linewidth=1)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.show()
        if (k == train_min and (epoch + 1) == 50):
            argmin = TotalLoss_epoch[epoch - 49:epoch].argmin()
            del model
            K.clear_session()
            time.sleep(2)
            print("Changing Optimizer to Adam")
            model = load_model(direction + "Window:_{}_Epoch:{}.h5".format(k, argmin), compile=False)
            sgd = optimizers.Adam(lr=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=3e-8, amsgrad=False)
            model = multi_gpu_model(model, gpus=2)
            model.compile(loss='mean_squared_error', optimizer=sgd)

    # Test after training all epochs
    argmin = TotalLoss_epoch.argmin()
    del model
    K.clear_session()
    print("Loading optimal model for testing")
    time.sleep(2)
    model = load_model(direction + "Window:_{}_Epoch:{}.h5".format(k, argmin), compile=False)
    sgd = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model = multi_gpu_model(model, gpus=2)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    SNP500 = pd.read_csv("./Chart_after_processing_US/SNP500_before_scaling_window{}.csv".format(k))
    SNP500 = SNP500.values
    Max_close = SNP500[:, 1].max()
    Min_close = SNP500[:, 1].min()
    mse, mape, mae, target, prediction = test_model(model, SNP500_input, SNP500_target, All_chart_info, \
                                                    Max_close, Min_close, mini_batch_size,
                                                    unit_iter * k + unit_iter * 4, unit_iter * k + unit_iter * 5)

    BH, MP = back_test(thresh_hold, transaction_cost, target, prediction)
    BH_profit.append(BH)
    Model_profit.append(MP)
    MSE.append(mse)
    MAPE.append(mape)
    MAE.append(mae)
    Arg_min.append(argmin)
    overall_prediction.append(prediction)
    overall_target.append(target)
    print("Window: {0:2d}, TestMSE: {1:.10f}".format(k, mse))
    result = {"MSE": MSE, "MAPE": MAPE, "MAE": MAE, "BH_Profit": BH_profit, "Model_Profit": Model_profit, \
              "Arg_min": Arg_min, "Arg_min_test": Arg_min_test}
    result = pd.DataFrame(result)
    result.to_csv(direction + "Result.csv", index=False)

overall_prediction = np.reshape(np.array(overall_prediction), (800))
overall_target = np.reshape(np.array(overall_target), (800))

mse, mape, mae = test_model_final(overall_prediction, overall_target, \
                                  mini_batch_size, unit_iter * train_min + unit_iter * 4,
                                  unit_iter * train_max + unit_iter * 4)

BH, MP = back_test(thresh_hold, transaction_cost, overall_target, overall_prediction)
BH_profit.append(BH)
Model_profit.append(MP)
MSE.append(mse)
MAPE.append(mape)
MAE.append(mae)
Arg_min.append(0)
Arg_min_test.append(0)
print("Window: {0:2d}, TestMSE: {1:.10f}".format(k, mse))
result = {"MSE": MSE, "MAPE": MAPE, "MAE": MAE, "BH_Profit": BH_profit, "Model_Profit": Model_profit, \
          "Arg_min": Arg_min}
result = pd.DataFrame(result)
result.to_csv(direction + "Result.csv", index=False)
np.savetxt(direction + "overall_prediction.csv".format(k), overall_prediction, delimiter=',')
np.savetxt(direction + "overall_target.csv".format(k), overall_target, delimiter=',')