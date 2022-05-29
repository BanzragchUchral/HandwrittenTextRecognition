from __future__ import division
from __future__ import print_function
import codecs
import sys
from word_beam_search import WordBeamSearch
import tensorflow.compat.v1 as tf



from DataLoader import FilePaths


class DecoderType:
    BestPath = 0
    WordBeamSearch = 1
    BeamSearch = 2


class Model:
    batchSize = 50
    imgSize = (800, 64) 
    maxTextLen = 100

    def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False):
        tf.disable_v2_behavior()
        tf.reset_default_graph()
        self.charList = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0

        self.inputImgs = tf.compat.v1.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))

        self.setupCNN()
        self.setupRNN()
        self.setupCTC()

        self.batchesTrained = 0
        self.learningRate = tf.compat.v1.placeholder(tf.float32, shape=[])
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

        (self.sess, self.saver) = self.setupTF()

        self.training_loss_summary = tf.compat.v1.summary.scalar('loss', self.loss)
        self.writer = tf.compat.v1.summary.FileWriter(
           './logs', self.sess.graph)  # Tensorboard: Create writer
        self.merge = tf.compat.v1.summary.merge([self.training_loss_summary])  # Tensorboard: Merge

    def setupCNN(self):

        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)

        with tf.compat.v1.name_scope('CONV1'):
            kernel = tf.Variable(
                tf.random.truncated_normal([5, 5, 1, 64], stddev=0.1))
            conv = tf.nn.conv2d(
                input=cnnIn4d, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool2d(input=learelu, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')

        with tf.compat.v1.name_scope('CONV2'):
            kernel = tf.Variable(tf.random.truncated_normal(
                [5, 5, 64, 128], stddev=0.1))
            conv = tf.nn.conv2d(
                input=pool, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool2d(input=learelu, ksize=(1, 1, 2, 1), strides=(1, 1, 2, 1), padding='VALID')

        with tf.compat.v1.name_scope('CONV3'):
            kernel = tf.Variable(tf.random.truncated_normal(
                [3, 3, 128, 128], stddev=0.1))
            conv = tf.nn.conv2d(
                input=pool, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            mean, variance = tf.nn.moments(x=conv, axes=[0])
            batch_norm = tf.nn.batch_normalization(
                conv, mean, variance, offset=None, scale=None, variance_epsilon=0.001)
            learelu = tf.nn.leaky_relu(batch_norm, alpha=0.01)
            pool = tf.nn.max_pool2d(input=learelu, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')

        with tf.compat.v1.name_scope('CONV4'):
            kernel = tf.Variable(tf.random.truncated_normal(
                [3, 3, 128, 256], stddev=0.1))
            conv = tf.nn.conv2d(
                input=pool, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)

        with tf.compat.v1.name_scope('CONV5'):
            kernel = tf.Variable(tf.random.truncated_normal(
                [3, 3, 256, 256], stddev=0.1))
            conv = tf.nn.conv2d(
                input=learelu, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool2d(input=learelu, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')

        with tf.compat.v1.name_scope('CONV6'):
            kernel = tf.Variable(tf.random.truncated_normal(
                [3, 3, 256, 512], stddev=0.1))
            conv = tf.nn.conv2d(
                input=pool, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            mean, variance = tf.nn.moments(x=conv, axes=[0])
            batch_norm = tf.nn.batch_normalization(
                conv, mean, variance, offset=None, scale=None, variance_epsilon=0.001)
            learelu = tf.nn.leaky_relu(batch_norm, alpha=0.01)
            pool = tf.nn.max_pool2d(input=learelu, ksize=(1, 1, 2, 1), strides=(1, 1, 2, 1), padding='VALID')

        with tf.compat.v1.name_scope('CONV7'):
            kernel = tf.Variable(tf.random.truncated_normal(
                [3, 3, 512, 512], stddev=0.1))
            conv = tf.nn.conv2d(
                input=pool, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            learelu = tf.nn.leaky_relu(conv, alpha=0.01)
            pool = tf.nn.max_pool2d(input=learelu, ksize=(1, 1, 2, 1), strides=(1, 1, 2, 1), padding='VALID')

            self.cnnOut4d = pool

    def setupRNN(self):
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

        numHidden = 512
        cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(
            num_units=numHidden, state_is_tuple=True, name='basic_lstm_cell') for _ in range(2)]
        stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        ((forward, backward), _) = tf.compat.v1.nn.bidirectional_dynamic_rnn(
            cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)

        concat = tf.expand_dims(tf.concat([forward, backward], 2), 2)

        kernel = tf.Variable(tf.random.truncated_normal(
            [1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
        self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])

    def setupCTC(self):
        self.ctcIn3dTBC = tf.transpose(a=self.rnnOut3d, perm=[1, 0, 2])

        with tf.compat.v1.name_scope('CTC_Loss'):
            self.gtTexts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[
                                           None, 2]), tf.compat.v1.placeholder(tf.int32, [None]), tf.compat.v1.placeholder(tf.int64, [2]))
            self.seqLen = tf.compat.v1.placeholder(tf.int32, [None])
            self.loss = tf.reduce_mean(input_tensor=tf.compat.v1.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen,
                               ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=True))
        with tf.compat.v1.name_scope('CTC_Decoder'):
            chars = ''.join(self.charList)
            wordChars = open(
                FilePaths.fnWordCharList).read().splitlines()[0]
            corpus = open(FilePaths.fnCorpus).read()

            self.decoder = WordBeamSearch(50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'),
                                          wordChars.encode('utf8'))

            self.wbs_input = tf.nn.softmax(self.ctcIn3dTBC, axis=2)

        # Return a CTC operation to compute the loss and CTC operation to decode the RNN output
        return self.loss, self.decoder

    def setupTF(self):
        """ Initialize TensorFlow """
        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)
        sess = tf.compat.v1.Session()  # Tensorflow session
        saver = tf.compat.v1.train.Saver(max_to_keep=3)  # Saver saves model to file
        modelDir = '../model/'
        latestSnapshot = tf.train.latest_checkpoint(modelDir)  # Is there a saved model?
        # If model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + modelDir)
        # Load saved model if available
        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)
        else:
            print('Init with new values')
            sess.run(tf.compat.v1.global_variables_initializer())

        return (sess, saver)

    def toSpare(self, texts):
        """ Convert ground truth texts into sparse tensor for ctc_loss """
        indices = []
        values = []
        shape = [len(texts), 0]  # Last entry must be max(labelList[i])
        # Go over all texts
        for (batchElement, texts) in enumerate(texts):
            # Convert to string of label (i.e. class-ids)
            print(texts)
            labelStr = []
            for c in texts:
                 print(c, '|', end='')
                 labelStr.append(self.charList.index(c))
            print(' ')
            labelStr = [self.charList.index(c) for c in texts]
            # Sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # Put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)

    def decoderOutputToText(self, ctcOutput):
        if self.decoderType == DecoderType.WordBeamSearch:
            encodedLabelStrs = ctcOutput

        else:

            decoded = ctcOutput[0][0]

            encodedLabelStrs = [[] for i in range(Model.batchSize)]

            idxDict = {b : [] for b in range(Model.batchSize)}
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0]
                encodedLabelStrs[batchElement].append(label)

        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]

    def trainBatch(self, batch, batchNum):
        sparse = self.toSpare(batch.gtTexts)
        rate = 0.001

        evalList = [self.merge, self.optimizer, self.loss]
        feedDict = {self.inputImgs: batch.imgs, self.gtTexts: sparse, self.seqLen: [Model.maxTextLen] * Model.batchSize, self.learningRate: rate}
        (loss_summary, _, lossVal) = self.sess.run(evalList, feedDict)

        self.writer.add_summary(loss_summary, batchNum)
        self.batchesTrained += 1
        return lossVal

    def return_rnn_out(self, batch, write_on_csv=False):
        numBatchElements = len(batch.imgs)
        decoded, rnnOutput = self.sess.run([self.decoder, self.ctcIn3dTBC],
                                {self.inputImgs: batch.imgs, self.seqLen: [Model.maxTextLen] * numBatchElements})

        decoded = rnnOutput
        print(decoded.shape)

        if write_on_csv:
            s = rnnOutput.shape
            b = 0
            csv = ''
            for t in range(s[0]):
                for c in range(s[2]):
                    csv += str(rnnOutput[t, b, c]) + ';'
                csv += '\n'
            open('mat_0.csv', 'w').write(csv)

        return decoded[:,0,:].reshape(100,80)

    def inferBatch(self, batch):
        numBatchElements = len(batch.imgs)

        eval_list = []

        if self.decoderType == DecoderType.WordBeamSearch:
            eval_list.append(self.wbs_input)
        else:
            eval_list.append(self.decoder)

        eval_list.append(self.ctcIn3dTBC)

        feedDict = {self.inputImgs: batch.imgs, self.seqLen: [Model.maxTextLen] * numBatchElements}
        evalRes = self.sess.run(eval_list, feedDict)

        if self.decoderType != DecoderType.WordBeamSearch:
            decoded = evalRes[0]
        else:
            decoded = self.decoder.compute(evalRes[0])

        texts = self.decoderOutputToText(decoded)
        return texts

    def save(self):
        """ Save model to file """
        self.snapID += 1
        self.saver.save(self.sess, '../model/snapshot',
                        global_step=self.snapID)

    def trainBatch(self, batch, batchNum):
        sparse = self.toSpare(batch.gtTexts)
        rate = 0.001
        evalList = [self.merge, self.optimizer, self.loss]
        feedDict = {self.inputImgs: batch.imgs, self.gtTexts: sparse, self.seqLen: [Model.maxTextLen] * Model.batchSize, self.learningRate: rate}
        (loss_summary, _, lossVal) = self.sess.run(evalList, feedDict)
        self.writer.add_summary(loss_summary, batchNum)
        self.batchesTrained += 1
        return lossVal

