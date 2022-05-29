from __future__ import division
from __future__ import print_function

import random
import os
import cv2
import numpy as np

from SamplePreprocessor import preprocessor


class FilePaths:
    """ Filenames and paths to data """
    fnCharList = '../data/charList.txt'
    fnWordCharList = '../data/wordCharList.txt'
    fnCorpus = '../data/corpus.txt'
    fnAccuracy = '../data/accuracy.txt'
    fnTrain = '../data/'
    fnInfer = '../data/testImage1.png'


class Sample:

    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath


class Batch:

    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts


class DataLoader:
    def __init__(self, filePath, batchSize, imgSize, maxTextLen, load_aug=True):

        assert filePath[-1] == '/'

        self.dataAugmentation = True # False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []

        f = open("../data/" + 'lines.txt')
        chars = set()
        bad_samples = []
        bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
        for line in f:
            if not line or line[0] == '#':
                continue

            lineSplit = line.strip().split(' ')

            fileNameSplit = lineSplit[0].split('-')

            fileName = filePath + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' +\
                       lineSplit[0] + '.png'

            if len(lineSplit) == 9:
                gtText_list = lineSplit[8].split('|')
            if len(lineSplit) > 9:
                l = len(lineSplit)
                for i in range(8, l - 1):
                    lineSplit[8] = lineSplit[8] + "|" + lineSplit[i + 1]
                    i = i + 1
                gtText_list = lineSplit[8].split('|')
            gtText = self.truncateLabel(' '.join(gtText_list), maxTextLen)
            chars = chars.union(set(list(gtText)))

            if not os.path.getsize(fileName):
                bad_samples.append(lineSplit[0] + '.png')
                continue

            self.samples.append(Sample(gtText, fileName))


        if set(bad_samples) != set(bad_samples_reference):
            print("Warning, damaged images found:", bad_samples)
            print("Damaged images expected:", bad_samples_reference)

        splitIdx = int(0.90 * len(self.samples))
        self.trainSamples = self.samples[:splitIdx]

        self.validateandtestSamples = self.samples[splitIdx:]
        splitIdy = int(0.5 * len(self.validateandtestSamples))

        self.validationSamples = self.validateandtestSamples[splitIdy:]
        self.testSamples = self.validateandtestSamples[:splitIdy]

        self.trainLines = [x.gtText for x in self.trainSamples]
        self.validationLines = [x.gtText for x in self.validationSamples]
        self.testLines = [x.gtText for x in self.testSamples]

        self.numTrainSamplesPerEpoch = 9500

        self.trainSet()

        self.charList = sorted(list(chars))

    def truncateLabel(self, text, maxTextLen):
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text


    def trainSet(self):
        self.dataAugmentation = True
        self.currIdx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples

    def validationSet(self):
        self.dataAugmentation = False
        self.currIdx = 0
        self.samples = self.validationSamples

    def getIteratorInfo(self):
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)

    def hasNext(self):
        return self.currIdx + self.batchSize <= len(self.samples)

    def getNext(self):
        batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        gtTexts = [self.samples[i].gtText for i in batchRange]
        imgs = [preprocessor(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize)
            for i in batchRange]
        self.currIdx += self.batchSize
        return Batch(gtTexts, imgs)
