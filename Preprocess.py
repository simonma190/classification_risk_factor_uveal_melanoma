import os.path

import MainRes
from MainRes import *

# allDataCSV path

# IMPORTANT: This program MUST ONLY BE RUN ONCE PER MODEL!
# It randomly selects the set of data to use as validation,
# if it is run multiple times for the same model the model will see parts of the validation set during training.
# This is very bad. If you want to use an old model and add new data you must manually add it to the correct csv

csv = 'UM Imaging Data/UM_ImageData.csv'

thickness = False
categorize = False
advancedDepth = False
subFluid = False
binaryfluid = True
margin = False
pigment = False
drusen = False
rpe = False
halo = False
hemorrhage = False
hollow = False
shape = False
modalities = ['OPTOS Fundus', 'Quantal US']
# size of the validation set (as a ratio of the total size)
val = .1

csv = pd.read_csv(csv, skiprows=[1], na_values=['-'])
# Query:
for modality in modalities:
    # data = csv[~csv[modality].isnull()]
    data = csv[~csv['OPTOS Fundus'].isnull()]
    data = data[~data['Quantal US'].isnull()]

    if advancedDepth:
        lr = 7
        savePath = 'advancedDepth'
        interval = 3
        offset = 0

        for i in np.arange(0, lr, interval):
            print(i)
            if i == (lr - 1):
                label = data[data['Lesion Thickness (mm)'] > (i - interval + offset)]
            elif i != 0:
                label = data[data['Lesion Thickness (mm)'] <= (i + offset)]
                label = label[label['Lesion Thickness (mm)'] > max((i - interval + offset), 0)]
            else:
                label = data[~data['OPTOS Control'].isnull()]
            if len(label != 0):
                test = label.sample(n=5, axis=0)
                train = label[~label.isin(test)]
                test.to_csv(os.path.join(savePath, modality + 'test' + str(i) + '.csv'))
                train.to_csv(os.path.join(savePath, modality + 'train' + str(i) + '.csv'))

    if thickness:
        savePath = 'thickness'
        label0 = (data[data['Lesion Thickness (mm)'] < 1])
        test0 = label0.sample(frac=val, axis=0)
        train0 = label0[~label0.isin(test0)]

        label1 = (data[data['Lesion Thickness (mm)'] < 2])
        label1 = label1[label1['Lesion Thickness (mm)'] >= 1]
        test1 = label1.sample(frac=val, axis=0)
        train1 = label1[~label1.isin(test1)]

        label2 = (data[data['Lesion Thickness (mm)'] >= 2])
        test2 = label2.sample(frac=val, axis=0)
        train2 = label2[~label2.isin(test2)]


        # saving
        test0.to_csv(os.path.join(savePath, modality + 'test0.csv'))
        test1.to_csv(os.path.join(savePath, modality + 'test1.csv'))
        test2.to_csv(os.path.join(savePath, modality + 'test2.csv'))
        train0.to_csv(os.path.join(savePath, modality + 'train0.csv'))
        train1.to_csv(os.path.join(savePath, modality + 'train1.csv'))
        train2.to_csv(os.path.join(savePath, modality + 'train2.csv'))

    if categorize:
        savePath = 'categorize'
        label1 = data[~data['OPTOS Control'].isnull()]
        test1 = label1.sample(frac=val, axis=0)
        train1 = label1[~label1.isin(test1)]

        train1.to_csv(os.path.join(savePath, 'controltrain.csv'))
        test1.to_csv(os.path.join(savePath, 'controltest.csv'))

        label2 = data[data['Diagnosis (0=control, 1=UM, 2=Nevus, 3=IMCT, 4=CHRPE, 5=Hemangioma, 6=Melanocytoma)'] == 2]

        test2 = label2.sample(frac=val, axis=0)
        train2 = label2[~label2.isin(test2)]

        train2.to_csv(os.path.join(savePath, modality + 'nevustrain.csv'))
        test2.to_csv(os.path.join(savePath, modality + 'nevustest.csv'))

        label3 = data[data['Diagnosis (0=control, 1=UM, 2=Nevus, 3=IMCT, 4=CHRPE, 5=Hemangioma, 6=Melanocytoma)'] == 1]

        test3 = label3.sample(frac=val, axis=0)
        train3 = label3[~label3.isin(test3)]

        train3.to_csv(os.path.join(savePath, modality + 'train.csv'))
        test3.to_csv(os.path.join(savePath, modality + 'test.csv'))

    if subFluid:
        savePath = 'subFluid'
        header = 'Subretinal Fluid (0=none, 1=apical, 2=<3mm from margin, 3=3-6mm from margin, 4=>6mm from margin; from Shields 2019)'
        lr = 5
        interval = 1
        offset = 0

        for i in np.arange(0, lr, interval):
            print(i)
            if i != 0:
                label = data[data[header] <= (i + offset)]
                label = label[label[header] > max((i - interval + offset), 0)]

            else:
                label = data[data[header] == 0]

            if len(label != 0):
                n = int(len(label) * val) + 1
                test = label.sample(n=n, axis=0)
                train = label[~label.isin(test)]
                test.to_csv(os.path.join(savePath, modality + 'test' + str(i) + '.csv'))
                train.to_csv(os.path.join(savePath, modality + 'train' + str(i) + '.csv'))

    if binaryfluid:
        savePath = 'binaryfluid'
        header = 'BINARY Subretinal Fluid (0=absent, 1=present)'
        lr = 2
        interval = 1
        offset = 0

        for i in np.arange(0, lr, interval):
            print(i)
            if i != 0:
                label = data[data[header] > 0]

            else:
                label = data[data[header] == 0]

            if len(label != 0):
                n = int(len(label) * val) + 1
                test = label.sample(n=n, axis=0)
                train = label[~label.isin(test)]
                test.to_csv(os.path.join(savePath, modality + 'test' + str(i) + '.csv'))
                train.to_csv(os.path.join(savePath, modality + 'train' + str(i) + '.csv'))

    if margin:
        savePath = 'margin'
        header = 'Margin to Optic Nerve Head (mm; -=unable to determine)'

        label0 = (data[data[header] < 3])
        test0 = label0.sample(frac=val, axis=0)
        train0 = label0[~label0.isin(test0)]


        label1 = data[data[header] >= 3]
        test1 = label1.sample(frac=val, axis=0)
        train1 = label1[~label1.isin(test1)]



        # saving
        test0.to_csv(os.path.join(savePath, modality + 'test0.csv'))
        test1.to_csv(os.path.join(savePath, modality + 'test1.csv'))

        train0.to_csv(os.path.join(savePath, modality + 'train0.csv'))
        train1.to_csv(os.path.join(savePath, modality + 'train1.csv'))

        # lr = 6
        # interval = 1
        # offset = 0
        #
        # for i in np.arange(0, lr, interval):
        #     print(i)
        #     if i == (lr - 1):
        #         label = data[data[header] > (i - interval + offset)]
        #     elif i != 0:
        #         label = data[data[header] <= (i + offset)]
        #         label = label[label[header] > max((i - interval + offset), 0)]
        #
        #     else:
        #         label = data[data[header] == 0]
        #
        #     if len(label != 0):
        #         n = int(len(label) * val) + 1
        #         test = label.sample(n=n, axis=0)
        #         train = label[~label.isin(test)]
        #         test.to_csv(os.path.join(savePath, modality + 'test' + str(i) + '.csv'))
        #         train.to_csv(os.path.join(savePath, modality + 'train' + str(i) + '.csv'))
    if pigment:
        savePath = 'pigment'
        header = 'BINARY Orange Pigment (0=none or indeterminate, 1=present or trace)'
        lr = 2
        interval = 1
        offset = 0

        for i in np.arange(0, lr, interval):
            print(i)
            label = data[data[header] == i]

            if len(label) != 0:
                n = int(len(label) * val) + 1
                test = label.sample(n=n, axis=0)
                train = label[~label.isin(test)]
                test.to_csv(os.path.join(savePath, modality + 'test' + str(i) + '.csv'))
                train.to_csv(os.path.join(savePath, modality + 'train' + str(i) + '.csv'))
    if drusen:
        savePath = 'drusen'
        header = 'BINARY Drusen (0=none or indeterminate, 1=present)'
        lr = 2
        interval = 1
        offset = 0

        for i in np.arange(0, lr, interval):
            print(i)
            label = data[data[header] == i]

            if len(label != 0):
                n = int(len(label) * val) + 1
                test = label.sample(n=n, axis=0)
                train = label[~label.isin(test)]
                test.to_csv(os.path.join(savePath, modality + 'test' + str(i) + '.csv'))
                train.to_csv(os.path.join(savePath, modality + 'train' + str(i) + '.csv'))
    if rpe:
        savePath = 'RPE'
        header = 'RPE changes  (0=none, 1=RPE changes, 2=fibrous metaplasia, 3=indeterminate)'
        lr = 3
        interval = 1
        offset = 0

        for i in np.arange(0, lr, interval):
            print(i)
            label = data[data[header] == i]

            if len(label != 0):
                n = int(len(label) * val) + 1
                test = label.sample(n=n, axis=0)
                train = label[~label.isin(test)]
                test.to_csv(os.path.join(savePath, modality + 'test' + str(i) + '.csv'))
                train.to_csv(os.path.join(savePath, modality + 'train' + str(i) + '.csv'))
    if halo:
        savePath = 'halo'
        header = 'Halo (0=none, 1=halo, 2=indeterminate)'
        lr = 2
        interval = 1
        offset = 0

        for i in np.arange(0, lr, interval):
            print(i)
            label = data[data[header] == i]

            if len(label != 0):
                n = int(len(label) * val) + 1
                test = label.sample(n=n, axis=0)
                train = label[~label.isin(test)]
                test.to_csv(os.path.join(savePath, modality + 'test' + str(i) + '.csv'))
                train.to_csv(os.path.join(savePath, modality + 'train' + str(i) + '.csv'))
    if hemorrhage:
        savePath = 'hemorrhage'
        header = 'BINARY Hemorrhage (0=none or indeterminate, 1=retina or vitreous hemorrhage present)'
        lr = 2
        interval = 1
        offset = 0

        for i in np.arange(0, lr, interval):
            print(i)
            label = data[data[header] == i]

            if len(label != 0):
                n = int(len(label) * val) + 1
                test = label.sample(n=n, axis=0)
                train = label[~label.isin(test)]
                test.to_csv(os.path.join(savePath, modality + 'test' + str(i) + '.csv'))
                train.to_csv(os.path.join(savePath, modality + 'train' + str(i) + '.csv'))
    if hollow:
        savePath = 'hollow'
        header = 'BINARY Hollow Ultrasound (0=none or indeterminate, 1=hollow)'
        lr = 2
        interval = 1
        offset = 0

        for i in np.arange(0, lr, interval):
            print(i)
            label = data[data[header] == i]

            if len(label != 0):
                n = int(len(label) * val) + 1
                test = label.sample(n=n, axis=0)
                train = label[~label.isin(test)]
                test.to_csv(os.path.join(savePath, modality + 'test' + str(i) + '.csv'))
                train.to_csv(os.path.join(savePath, modality + 'train' + str(i) + '.csv'))
    if shape:
        savePath = 'shape'
        header = 'BINARY Mushroom Shape (0=flat or other, 1=mushroom)'
        lr = 2
        interval = 1
        offset = 0

        for i in np.arange(0, lr, interval):
            print(i)
            label = data[data[header] == i]

            if len(label) >= 2:
                n = int(len(label) * val) + 1
                test = label.sample(n=n, axis=0)
                train = label[~label.isin(test)]
                test.to_csv(os.path.join(savePath, modality + 'test' + str(i) + '.csv'))
                train.to_csv(os.path.join(savePath, modality + 'train' + str(i) + '.csv'))
