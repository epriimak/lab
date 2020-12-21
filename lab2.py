import numpy as np
import matplotlib.pyplot as plt
import math


def read_signal(fname: str, one_signal_size: int):
    data = []
    with open(fname, 'r') as inf:
        for line in inf.readlines():
            remove_dirst_str = line.replace("[", "")
            remove_next_str = remove_dirst_str.replace("]", "")
            data.append(remove_next_str.split(", "))

    data_float_format = []
    for item in data:
        data_float_format.append([float(x) for x in item])

    new_data = np.asarray(data_float_format)
    data = np.reshape(new_data, (new_data.shape[1] // one_signal_size, one_signal_size))
    return data


def get_array(array):
    ind = [obj for obj in range(len(array))]
    return ind


def draw_signal(signal, ind, title):
    plt.title(title)
    plt.plot(ind, signal)
    plt.show()


def get_areas(signal):
    bin = int(math.log2(len(signal) + 1))
    hist = plt.hist(signal, bins=bin)
    plt.show()

    count = []
    start = []
    finish = []
    types = [0] * bin

    for i in range(bin):
        count.append(hist[0][i])
        start.append(hist[1][i])
        finish.append(hist[1][i + 1])

    sortedHist = sorted(count)
    repeat = 0
    for i in range(bin):
        for j in range(bin):
            if sortedHist[len(sortedHist) - 1 - i] == count[j]:
                if repeat == 0:
                    types[j] = "фон"
                elif repeat == 1:
                    types[j] = "сигнал"
                else:
                    types[j] = "переход"
                repeat += 1

    return start, finish, types


def get_converted_data(signal, indArray, start, finish, types):
    p_type = [0] * len(indArray)
    zones = []
    areas_type = []

    for i in range(len(indArray)):
        for j in range(len(types)):
            if (signal[i] >= start[j]) and (signal[i] <= finish[j]):
                p_type[i] = types[j]

    currType = p_type[0]
    start = 0
    for i in range(len(p_type)):
        if currType != p_type[i]:
            finish = i
            areas_type.append(currType)
            zones.append([start, finish])
            start = finish
            currType = p_type[i]

    if currType != areas_type[len(areas_type) - 1]:
        areas_type.append(currType)
        zones.append([finish, len(indArray) - 1])

    return zones, areas_type


def reduce_emissions(signal, indArray, area_data, types):
    while len(types) > 5:
        for i in range(len(types)):
            if (types[i] == "переход") and (types[i - 1] == types[i + 1]):
                startValue = signal[area_data[i - 1][1] - 1]
                finishValue = signal[area_data[i + 1][0] + 1]
                newValue = (startValue + finishValue) / 2
                num = area_data[i][1] - area_data[i][0]
                for j in range(num):
                    signal[area_data[i][0] + j] = newValue

        start, finish, types = get_areas(signal)
        area_data, types = get_converted_data(signal, indArray, start, finish, types)

    return signal, area_data, types


def draw_areas(signal, indArray, area_data, types, title):
    plt.title(title)
    for i in range(len(area_data)):
        if types[i] == "фон":
            color_ = 'y'
        if types[i] == "сигнал":
            color_ = 'r'
        if types[i] == "переход":
            color_ = 'g'

        plt.plot(indArray[area_data[i][0]:area_data[i][1]],
                 signal[area_data[i][0]:area_data[i][1]], color=color_, label=types[i])
    plt.legend()
    plt.show()


def get_inter_group_D(signal):
    summ = 0.0
    mean = np.empty(signal.shape[0])
    for i in range(len(signal)):
        mean[i] = np.mean(signal[i])
    meanMean = np.mean(mean)

    for i in range(len(mean)):
        summ += (mean[i] - meanMean) ** 2
    summ /= (signal.shape[0] - 1)

    return len(signal) * summ


def get_intar_group_D(signal):
    result = 0.0
    for i in range(signal.shape[0]):
        mean = np.mean(signal[i])
        summ = 0.0
        for j in range(signal.shape[1]):
            summ += (signal[i][j] - mean) ** 2
        summ /= (signal.shape[0] - 1)
        result += summ

    return result / signal.shape[0]


def get_F(signal, k):
    newSizeY = int(signal.size / k)
    newSizeX = k
    print("k = " + str(k))
    splitData = np.reshape(signal, (newSizeX, newSizeY))
    interGroup = get_inter_group_D(splitData)
    print("Inter = " + str(interGroup))
    intraGroup = get_intar_group_D(splitData)
    print("Intar = " + str(intraGroup))
    print("F = " + str(interGroup / intraGroup))
    return interGroup / intraGroup


def get_K(num):
    i = 4
    while num % i != 0:
        i += 1
    return i


def get_Fisher(signal, area_data):
    fishers = []
    for i in range(len(area_data)):
        start = area_data[i][0]
        finish = area_data[i][1]
        k = get_K(finish - start)
        while k == finish - start:
            finish += 1
            k = get_K(finish - start)
        fishers.append(get_F(signal[start:finish], int(k)))
    return fishers


fileData = read_signal("wave_ampl.txt", 1024)
idSignal = 543
indArray = get_array(fileData[idSignal])
signal = fileData[idSignal]

draw_signal(signal, indArray, "Сигнал " + str(idSignal))

start, finish, types = get_areas(signal)
area_data, types = get_converted_data(signal, indArray, start, finish, types)
copySignal = signal.copy()
signalWithoutEmissions, area_data, types = reduce_emissions(copySignal, indArray, area_data, types)

draw_signal(signalWithoutEmissions, indArray, "Сигнал " + str(idSignal) + " без выбросов ")
#draw_areas(signal, indArray, area_data, types, "Разделение областей для входного сигнала")
draw_areas(signalWithoutEmissions, indArray, area_data, types, "Разделенные области для сигнала без выбросов")

fishers = get_Fisher(signal, area_data)
print(fishers)
#print(get_F(signal, 32))
