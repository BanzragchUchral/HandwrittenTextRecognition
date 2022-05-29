import argparse
import os
import sys

import cv2
import editdistance
import numpy as np
import tensorflow.compat.v1 as tf

from DataLoader import Batch, DataLoader, FilePaths
from SamplePreprocessor import preprocessor, wer
from Model import DecoderType, Model

def load_different_image():
    imgs = []
    for i in range(1, 20):
       imgs.append(preprocessor(cv2.imread("../data/check_image/a ({}).png".format(i), cv2.IMREAD_GRAYSCALE), Model.imgSize, enhance=False))
    return imgs


def generate_random_images():
    return np.random.random((Model.batchSize, Model.imgSize[0], Model.imgSize[1]))


def infer(model, fnImg):
    img = preprocessor(fnImg, imgSize=Model.imgSize)
    if img is None:
        print("Image not found")

    imgs = load_different_image()
    imgs = [img] + imgs
    batch = Batch(None, imgs)
    recognized = model.inferBatch(batch)

    return recognized[0]


def predict(images):
    decoderType = DecoderType.WordBeamSearch

    model = Model(open(FilePaths.fnCharList).read(),
                  decoderType, mustRestore=False)

    texts = []
    for image in images:
        text = infer(model, image)
        texts.append(text)
    return texts


'''
def validate(model, loader):
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0

    totalCER = []
    while loader.hasNext():
        batch = loader.getNext()
        recognized = model.inferBatch(batch)

        for i in range(len(recognized)):
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            currCER = dist/max(len(recognized[i]), len(batch.gtTexts[i]))
            totalCER.append(currCER)

            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])

    charErrorRate = sum(totalCER)/len(totalCER)
    return charErrorRate


def train(model, loader):
    epoch = 0
    bestCharErrorRate = float('inf')
    noImprovementSince = 0
    earlyStopping = 10
    batchNum = 0

    totalEpoch = len(loader.trainSamples)//Model.batchSize
    f = open("lossfile.txt", "a")
    g = open("cerfile.txt", "a")

    while True:
        epoch += 1

        epochloss = ""
        loader.trainSet()
        while loader.hasNext():
            batchNum += 1
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch, batchNum)
            epochloss = loss
            print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

        charErrorRate = validate(model, loader)

        if charErrorRate < bestCharErrorRate:
            f.write(str(epochloss) + ",")
            g.write(str(charErrorRate) + ",")
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        if noImprovementSince >= earlyStopping:
            f.close()
            g.close()
            print('No more improvement since %d epochs. Training stopped.' %
                  earlyStopping)
            break
'''
