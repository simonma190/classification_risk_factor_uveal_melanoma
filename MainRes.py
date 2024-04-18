import random

import numpy as np
import pandas
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torchvision.io import read_image
import pathlib
from pathlib import Path
import imageio as iio
import matplotlib.pyplot as plt
import os
from itertools import cycle
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
import sys
import scipy.stats
from scipy.stats import norm, ttest_ind, wilcoxon, bootstrap, sem, iqr
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report


# Use the torchvision's implementation of ResNeXt, but add FC layer for a different number of classes (27) and a
# Sigmoid instead of a default Softmax. This is for loading the model and changing the last layer to allow multiple
# classifications (sigmoid instead of softmax)
class Resnet18(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))


def errorplot(data: pd.DataFrame):
    dataset = data.sort_values('AUC')
    for AUC, lower, upper, y in zip(dataset['AUC'], dataset['lower_ci'], dataset['upper_ci'], range(len(dataset))):
        plt.plot((AUC, lower, upper), (y, y, y), '|-', color='blue')
    plt.yticks(range(len(dataset)), dataset['category'])
    plt.axvline(x=0.5, linestyle='dotted', color='black')

    # Add x-axis label
    plt.xlabel('ROC AUC')

    # Show plot
    plt.show()


def write_classification_report_to_csv(report_dict, file_path):
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(file_path)


def mean_confidence_interval(data, alpha=0.95):
    # --------------------------------------------------------------------------
    # Computes confidence interval around mean.
    # --------------------------------------------------------------------------

    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(data, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(data, p))

    m = np.mean(data)

    return m, lower, upper


def compute_roc_auc(predictions, ground_truth):
    n_classes = predictions.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    mu = dict()
    lower = dict()
    upper = dict()

    if n_classes <= 2:
        fpr[0], tpr[0], _ = roc_curve(ground_truth[:, -1], predictions[:, -1])
        roc_auc[0] = auc(fpr[0], tpr[0])
        aucs = bootstrap_auc(predictions[:, -1], ground_truth[:, -1])
        # Compute confidence interval around mean, and the standard error
        mu[0], lower[0], upper[0] = mean_confidence_interval(aucs, .95)
    else:
        # Compute ROC area for each class (one vs all)
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(ground_truth[:, i], predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            aucs = bootstrap_auc(predictions[:, i], ground_truth[:, i])
            # Compute confidence interval around mean, and the standard error
            mu[i], lower[i], upper[i] = mean_confidence_interval(aucs, .95)

        # Compute ROC area for the micro-average, this is dubious for 2 class classification according to this thread:
        # https://datascience.stackexchange.com/questions/89180/c  ban-micro-average-roc-auc-score-be-larger-than-class-roc-auc-scores
        fpr["micro"], tpr["micro"], _ = roc_curve(ground_truth.ravel(), predictions.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        aucs = bootstrap_auc(predictions.ravel(), ground_truth.ravel())
        mu["micro"], lower["micro"], upper["micro"] = mean_confidence_interval(aucs, .95)

    return fpr, tpr, roc_auc, mu, lower, upper


def bootstrap_auc(p, q, n_bootstraps=1000):
    aucs = []

    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        ix = np.sort(np.random.randint(0, len(p), int(len(p) / 2)))
        try:
            aucs.append(roc_auc_score(q[ix], p[ix]))
        except ValueError:
            pass
    aucs = np.sort(aucs)
    return aucs


def plot_roc_curve(fpr, tpr, auc_score, mu, lower, upper, class_names=None, save_path=None, modality=None):
    # Set up the plot
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')

    # Plot the ROC curve and AUC for each class
    if class_names is None:
        class_names = ['Class ' + str(i) for i in range(len(fpr))]
    colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black'])
    for i, color in zip(range(len(fpr) - 1), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {auc_score[i]:.2f}, 95% CI = {lower[i]:.3f} - {upper[i]:.3f}')

    try:
        # Plot the micro-average ROC curve and AUC
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f' Micro: (AUC = {auc_score["micro"]:.2f}, 95% CI = {lower["micro"]:.3f} - {upper["micro"]:.3f})',
                 linestyle='--')
    except:
        for i, color in zip(range(len(fpr)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{class_names[i]} (AUC = {auc_score[i]:.2f}, 95% CI = {lower[i]:.3f} - {upper[i]:.3f}')

    plt.legend()
    #
    # plt.show(block=False)
    # plt.pause(.001)
    # Save the plot if a save_path is provided
    if save_path is not None:
        plt.savefig(os.path.join(save_path, modality))


# image Loader:
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, csvs, imgdir, rootdir, transform=None, toRGB=False, dataColumns=[1], scale=False):
        self.csvs = csvs
        length = 0
        fullcsvs = []
        csvlengths = []
        fulldata = pd.DataFrame(columns=["dir", "index"])
        for csv in csvs:
            csv_info = pd.read_csv(os.path.join(rootdir, csv))
            for column in dataColumns:
                col = pd.DataFrame(data={'dir': csv_info[csv_info.columns[column]]})
                col = col.dropna(how="all")
                indexcol = np.empty(len(col))
                indexcol.fill(column)
                col['index'] = indexcol.tolist()
                fulldata = pd.concat((fulldata, col))
            length += len(fulldata)
            csvlengths.append(length)
            fullcsvs.append(fulldata)
            fulldata = pd.DataFrame(columns=["dir", "index"])

        self.csvlengths = csvlengths
        self.fullcsvs = fullcsvs
        self.length = length
        self.dataColumns = dataColumns
        self.csv_info = fulldata
        self.rootdir = rootdir
        self.imgdir = imgdir
        self.transform = transform
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.toRGB = toRGB
        self.scale = scale

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tempindicies = [i for i, element in enumerate(self.csvlengths) if element > idx]
        tempindex = tempindicies[0]

        if tempindex > 0:
            trueidx = idx - self.csvlengths[tempindex - 1]
        else:
            trueidx = idx

        csv = self.fullcsvs[tempindex]

        row = csv.iloc[trueidx]

        img_path = os.path.join(self.imgdir, row[0])

        image = iio.imread(img_path + ".jpg")

        # converts a black and white image to RGB format by copying it three times
        if self.toRGB:
            image = np.repeat(image[..., np.newaxis], 3, -1)

        if self.transform:
            image = self.transform(image)

        category = tempindex
        if self.scale:
            tempcat = [1] * tempindex
            category = ((tempindicies[-1] - len(tempcat)) * [0]) + tempcat
        category = torch.tensor(np.array(category))
        category = category.type(torch.LongTensor)

        image = image.to(self.device)
        category = category.to(self.device)
        return image, category


def train_loop(dataloader, model, loss_fn, optimizer, nout, scale):  # train loop
    size = len(dataloader.dataset)
    correct = 0
    test_loss = 0
    tcorrect = 0
    current = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # As in tes_loop, the scale stuff is all broken here

        # if not scale:
        #     y = torch.reshape(y, [X.shape[0],nout])

        pred = model(X)

        y = torch.nn.functional.one_hot(y, pred.shape[1])

        if scale:
            # print("Train True Value: ", y, "Predicted: ",
            #       pred)
            loss = loss_fn(pred, y.type(torch.float))
        else:
            # print("True Value: ", y.type(torch.float32).sum().item(), "Predicted: ",
            #       pred.argmax(1).type(torch.float32).sum().item())

            # print("Train True Value: ", y.argmax(1), "Predicted: ",
            #       pred.argmax(1))
            # print(y.dtype, pred.dtype)
            loss = loss_fn(pred, y.type(torch.float))
        # get the loss
        batchloss = loss.item()
        test_loss += batchloss
        # gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate the number of correct predictions
        if scale:
            for i, item in enumerate(pred):
                tc = 1
                for j, n in enumerate(item):
                    if round(float(n)) == int(y[i][j]):
                        correct += 1
                    else:
                        tc = 0
                tcorrect += tc
        else:
            tcorrect += (pred.argmax(1) == y.argmax(1)).type(torch.float32).sum().item()
            correct = tcorrect
        # update the number of images seen so far
        current += len(X)
        if batch % 6 == 0:  # print the loss and accuracy every 6 batches
            avg_loss = test_loss / current  # average loss, sum of losses/number of images seen
            pcorrect = correct / current  # average accuracy, number of correct predictions/number of images seen * number of outputs
            tpc = tcorrect / current  # average "true accuracy," number of correct scale predictions/number of images
            # seen
            print(
                f"AvgLoss: {avg_loss:>7f} [{current:>5d}/{size:>5d}] AvgAccuracy: {(100 * pcorrect):>0.2f}% Correct: {correct:>.0f}")
            sys.stdout.flush()

    avg_loss = test_loss / len(dataloader)
    pcorrect = correct / size
    print(f"AvgLoss: {avg_loss:>7f} AvgAccuracy: {(100 * pcorrect):>0.2f}% "
          f"TrueAccuracy: {(100 * tpc):>0.2f}%")
    return test_loss / len(dataloader), model


def tes_loop(dataloader, model, loss_fn, nout, scale,
             savepath=None, modality=None):  # test loop, returns loss and if savepath is defined saves roc curve
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    tcorrect = 0
    model.eval()

    # with torch.no_grad():  # no need to calculate gradients
    for batch, (X, y) in enumerate(dataloader):  # iterate through batches

        pred = model(X)  # get predictions
        # move to cpu
        pred = pred.to('cpu')
        y = y.to('cpu')
        # convert ground_truth categories to one hot
        y = torch.nn.functional.one_hot(y, pred.shape[1])

        # creates prediction and ground truth arrays for roc curve
        if batch == 0:
            last_batch_size = pred.shape[0]
            allpreds = torch.empty(size=(size, pred.shape[1]))
            alltrue = torch.empty(size=(size, pred.shape[1]))
        allpreds[batch * last_batch_size:batch * last_batch_size + pred.shape[0], :] = torch.nn.functional.softmax(
            pred, dim=1)
        alltrue[batch * last_batch_size:batch * last_batch_size + pred.shape[0], :] = y
        last_batch_size = pred.shape[0]

        # This is broken if scale is true, need to fix
        if scale:
            print("True Value: ", y, "Predicted: ",
                  pred)
            loss = loss_fn(pred, y.type(torch.float))
        else:
            print("Test True Value: ", y.argmax(1), "Predicted: ",
                  pred.argmax(1))
            loss = loss_fn(pred, y.type(torch.float))

            # Grad cam code
        if savepath is not None:
            target_layers = [model.layer4[-1]]

            input_tensor = X

            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
            targets = [ClassifierOutputTarget(y.argmax().item())]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            nicex = X.to('cpu')
            nicex = np.array(nicex)
            nicex = np.transpose(nicex)
            nicex = np.reshape(nicex, (512, 512, 3))
            nicex = np.transpose(nicex, [1, 0, 2])

            visualization = show_cam_on_image(nicex, grayscale_cam, use_rgb=True)
            plt.cla()
            plt.clf()
            f = plt.figure()
            f.add_subplot(1, 2, 1)
            plt.imshow(visualization)
            f.add_subplot(1, 2, 2)
            plt.imshow(nicex)
            plt.pause(.001)
            if (pred.argmax(1) != y.argmax(1)):
                plt.xlabel("Failed to Predict", fontsize=18)
            plt.savefig(os.path.join(savepath, modality + str(batch)))
            plt.close(f)

        batchloss = loss.item()  # get the loss
        test_loss += batchloss  # add to total loss

        # need to revisit this is scale is true
        if scale:
            for i, item in enumerate(pred):
                tc = 1
                for j, n in enumerate(item):
                    if round(float(n)) == int(y[i][j]):
                        correct += 1
                    else:
                        tc = 0
                tcorrect += tc
        else:
            tcorrect += (pred.argmax(1) == y.argmax(1)).type(torch.float32).sum().item()
            correct = tcorrect

    # Convert the PyTorch tensors to numpy arrays
    predictions = allpreds.cpu().detach().numpy()
    labels = alltrue.cpu().detach().numpy()

    # Remove the extra columns
    predictions = predictions[:, 0:nout]
    labels = labels[:, 0:nout]
    roc_auc = None
    # Compute the ROC AUC if savepath is defined. This is a sort of dirty way to do this
    if (savepath is not None):
        fpr, tpr, roc_auc, mu, lower, upper = compute_roc_auc(predictions, labels)
        plot_roc_curve(fpr, tpr, roc_auc, save_path=savepath, modality=modality, mu=mu, lower=lower, upper=upper)
        np.savetxt(os.path.join(savepath, 'predictions.csv'), predictions, delimiter=',')
        np.savetxt(os.path.join(savepath, 'labels.csv'), labels, delimiter=',')
    else:
        fpr, tpr, roc_auc, mu, lower, upper = None, None, None, None, None, None

        # Compute the accuracy, true accuracy, and average loss. "True accuracy" is intended for non-binary
    # classification as the percentage of images that classified exactly correct ([0,1,1] == [0,1,1] but not [0,1,0])
    ncorrect = correct
    test_loss /= num_batches
    correct /= size
    tcorrect /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, \n True Accuracy: {(100 * tcorrect):>0.1f}%, Avg loss: {test_loss:>8f} \n nCorrect: {ncorrect} \n AUC: {roc_auc}")
    return test_loss, savepath, modality, tpr, fpr, roc_auc, lower, upper, labels, predictions


margin = True
pigment = True
hemorrhage = True
shape = True

if __name__ == '__main__':
    # import matplotlib
    # matplotlib.use("Agg") #This prevents a memory error from generating too many plots but turns off the gui
    # this is set up to iterate through all the different rootdirs and image modalities, rootdirs are named after the
    # categories
    rootdirs = ['binaryfluid','thickness','categorize','subFluid','binaryfluid','margin','pigment','drusen','rpe','halo','hemorrhage', 'hollow', 'shape']
    modalities = ['Quantal US', 'OPTOS Fundus']
    # ,
    alldata = pd.DataFrame(columns=['category', 'AUC', 'lower_ci', 'upper_ci'])
    for rootdir in rootdirs:
        for modality in modalities:
            print(modality, rootdir)
            torch.cuda.empty_cache()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using {device} device")

            #########################
            # MODIFIABLE PARAMETERS #
            #########################

            imagedir = "Images"

            scale = False  # Intended to account for non-binary classifiers, for now I have taken this out so it may
            # break if True. If you want to use it, you'll need to adjust the training loop and test loop to account for
            # the change. The image loader should work fine (it will load as [0,1,1] for 2)

            loss_fn = nn.CrossEntropyLoss()  # How should loss be calculated

            train = False #Are you training the model or testing?

            imgtype = ".png"  # the file format of the images you're loading

            batch_size = 30  # how many images the training processes at once
            if train == False:
                batch_size = 1 #Gradcam cannot handle multiple images at once


            # The following are irrelevant if train = False
            learning_rate = .00005  # Learning rate, between 0 and 1
            regularSave = False  # Should the network save a copy of itself at regular intervals
            saveFrequency = 10  # if above is true, how regular
            saveUniqueModels = False  # Should regular saves have unique names or should they override each other
            saveBestPerformers = True  # Should the model keep a copy of the best version (based on validation loss)
            if train:
                epochs = 50  # How many times should it train on the same set of data
            else:
                epochs = 1

            bestLoss = 9  # Maximum loss to save. Set high unless you are retraining a model
            maxcatagories = 9  # max number of categories, note this can be arbitrarily high but will slow down training
            # --------------------------#

            interval = 1
            testpath = []
            trainpath = []

            # Prep the paths for pulling data:
            ##Special case for categorization
            if rootdir == 'categorize':
                # testpath.append('controltest.csv')
                testpath.append(modality + 'nevustest.csv')
                testpath.append(modality + 'test.csv')

                # trainpath.append('controltrain.csv')
                trainpath.append(modality + 'nevustrain.csv')
                trainpath.append(modality + 'train.csv')
                nout = len(testpath)
                if modality == 'Quantal US':
                    datacolumns = [11]
                else:
                    datacolumns = [3]
            else:  # All other cases, expecting test + str(i) + .csv and train + str(i) + .csv as file names within the
                # roodir subfolder with the modality name (ex: thickness/Quantal UStrain0.csv)
                for i in range(0, maxcatagories, interval):
                    try:
                        csv = pd.read_csv(os.path.join(rootdir, modality + 'test' + str(i) + '.csv'))
                        csv = pd.read_csv(os.path.join(rootdir, modality + 'train' + str(i) + '.csv'))
                        testpath.append(modality + 'test' + str(i) + '.csv')
                        trainpath.append(modality + 'train' + str(i) + '.csv')
                    except:
                        print('csv: ' + modality + 'train' + str(i) + '.csv' + ' not found')
                nout = len(testpath)
                if modality == 'Quantal US':
                    datacolumns = [11]
                else:
                    datacolumns = [3]

            # Uncomment the next lines if training a new model:
            # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
            # model.fc = nn.Sequential(
            #     nn.Dropout(p=0.3),
            #     nn.Linear(in_features=model.fc.in_features, out_features=nout)
            # )

            # # Uncomment the next line if loading an existing model (bvmodel.pt is the 'best variance' model, but you can load any model here)
            model = torch.load(os.path.join(rootdir, modality + 'bvmodel.pt'))

            # transformations for training data
            transform = transforms.Compose(
                [ToTensor(), transforms.RandomHorizontalFlip(.5), transforms.RandomRotation(10), Resize((1024, 1024)),
                 transforms.RandomCrop(800), Resize((512, 512))])

            testDataset = CustomImageDataset(csvs=testpath, rootdir=rootdir, imgdir=imagedir,
                                             transform=transforms.Compose([ToTensor(), Resize((512, 512))]),
                                             toRGB=False,
                                             dataColumns=datacolumns, scale=scale)

            trainDataset = CustomImageDataset(csvs=trainpath, rootdir=rootdir, imgdir=imagedir,
                                              transform=transform,
                                              toRGB=False,
                                              dataColumns=datacolumns, scale=scale)

            net = model.to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

            testdataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=0)
            traindataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=0)

            modelData = pd.DataFrame(columns=['TrainLoss', 'TestLoss'], index=range(epochs))
            # plt.ion()

            for t in range(epochs):

                x = range(t + 1)
                print(f"{rootdir}\t {modality} \n Epoch {t + 1}\n-------------------------------")
                if train:
                    plt.clf()
                    trainLoss, net = train_loop(traindataloader, net, loss_fn, optimizer, nout, scale)
                    testLoss, _, _, _, _, _, _, _, _, _ = tes_loop(testdataloader, net, loss_fn, nout, scale)
                    modelData.iat[t, 0] = trainLoss
                    y1 = modelData.iloc[:(t + 1), 0]
                    if t != 0:
                        plt.close()
                    plt.plot(x, y1, label="Train")
                    if saveUniqueModels:
                        tag = ""
                    else:
                        tag = t
                    if (regularSave and t + 1 % saveFrequency == 0) or (saveBestPerformers and testLoss < bestLoss):
                        if saveBestPerformers and testLoss < bestLoss:
                            print("Saving new best, with a validation loss of: ", testLoss)
                            bestLoss = testLoss
                            torch.save(net, os.path.join(rootdir, modality + "bvmodel.pt"))
                        else:
                            print("Saving model: ", t)
                            name = "model" + tag + ".pt"
                            torch.save(net, os.path.join(rootdir, modality + name))

                    modelData.iat[t, 1] = testLoss

                    y2 = modelData.iloc[:(t + 1), 1]
                    plt.plot(x, y2, label="Test")
                    plt.ylabel("Loss", fontsize=18)
                    plt.xlabel("Epoch", fontsize=18)
                    plt.legend()
                    # plt.show()
                    # plt.pause(.01)
                else:
                    testLoss, path, m, tpr, fpr, roc_auc, lower, upper, labels, predictions = tes_loop(testdataloader,
                                                                                                       net, loss_fn,
                                                                                                       nout, scale,
                                                                                                       rootdir,
                                                                                                       modality)
                    for key in roc_auc:
                        if (len(roc_auc)==1):
                            temp = [modality +" "+ rootdir, roc_auc[key], lower[key], upper[key]]
                        else:
                            temp = [modality +" " + rootdir + " " + str(key), roc_auc[key], lower[key], upper[key]]
                        alldata.loc[len(alldata)] = temp
                    report = classification_report(labels.argmax(1), predictions.argmax(1), output_dict=True)
                    print(report)

                    write_classification_report_to_csv(report,
                                                       str(os.path.join(rootdir, modality + rootdir + '_report.csv')))
                    lab = pd.DataFrame(labels)
                    pred = pd.DataFrame(predictions)
                    lab.to_csv(os.path.join(rootdir, modality + "label.csv"))
                    pred.to_csv(os.path.join(rootdir, modality + "predictions.csv"))

            print("Done!")
            alldata.to_csv("stats.csv")
            plt.clf()
    errorplot(alldata)
