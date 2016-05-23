import numpy
import scipy.io.wavfile
import scikits.audiolab

import random
import time

def get_data(data_path, n_files, BATCH_SIZE, SEQ_LEN, OVERLAP, Q_LEVELS, Q_ZERO):
    """
    Generator that yields training inputs (subbatch, reset). `subbatch` contains
    quantized audio data; `reset` is a boolean indicating the start of a new
    sequence (i.e. you should reset h0 whenever `reset` is True).

    Feeds subsequences which overlap by a specified amount, so that the model
    can always have target for every input in a given subsequence.

    Loads sequentially-named FLAC files in a directory
    (p0.flac, p1.flac, p2.flac, ..., p[n_files-1].flac)

    Assumes all flac files have the same length.

    data_path: directory containing the flac files
    n_files: how many FLAC files are in the directory
    (see two_tier.py for a description of the constants)

    returns: (subbatch, reset)
    subbatch.shape: (BATCH_SIZE, SEQ_LEN + OVERLAP)
    reset: True or False
    """

    def round_to(x, y):
        """round x up to the nearest y"""
        return int(numpy.ceil(x / float(y))) * y

    def batch_quantize(data):
        """
        floats in (-1, 1) to ints in [0, Q_LEVELS-1]
        scales normalized across axis 1
        """
        eps = numpy.float64(1e-5)
        data -= data.min(axis=1)[:, None]
        data *= ((Q_LEVELS - eps) / data.max(axis=1)[:, None])
        data += eps/2
        data = data.astype('int32')
        return data%Q_LEVELS

    paths = [data_path+'/p{}.flac'.format(i) for i in xrange(n_files)]

    random.seed(123)
    random.shuffle(paths)

    batches = []
    for i in xrange(len(paths) / BATCH_SIZE):
        batches.append(paths[i*BATCH_SIZE:(i+1)*BATCH_SIZE])

    random.shuffle(batches)

    out = []

    SEQ_LEN += 4

    for batch_paths in batches:
        # batch_seq_len = length of longest sequence in the batch, rounded up to
        # the nearest SEQ_LEN.
        batch_seq_len = len(scikits.audiolab.flacread(batch_paths[0])[0])
        batch_seq_len = round_to(batch_seq_len, SEQ_LEN)

        batch = numpy.zeros(
            (BATCH_SIZE, batch_seq_len), 
            dtype='float32'
        )

        for i, path in enumerate(batch_paths):
            data, fs, enc = scikits.audiolab.flacread(path)
            batch[i, :] = data[:batch_seq_len]

        batch = batch_quantize(batch)

        batch = numpy.concatenate([
            numpy.full((BATCH_SIZE, OVERLAP), Q_ZERO, dtype='int32'),
            batch
        ], axis=1)

        for i in xrange(batch.shape[1] / SEQ_LEN):
            reset = numpy.int32(i==0)
            subbatch = batch[:, i*SEQ_LEN : (i+1)*SEQ_LEN]
#            out1.append(subbatch)
#            out2.append(reset)
            out.append((subbatch,reset))
    return out

def get_test_data(data_path, n_files, BATCH_SIZE, SEQ_LEN, OVERLAP, Q_LEVELS, Q_ZERO):
    """
    Generator that yields training inputs (subbatch, reset). `subbatch` contains
    quantized audio data; `reset` is a boolean indicating the start of a new
    sequence (i.e. you should reset h0 whenever `reset` is True).

    Feeds subsequences which overlap by a specified amount, so that the model
    can always have target for every input in a given subsequence.

    Loads sequentially-named FLAC files in a directory
    (p0.flac, p1.flac, p2.flac, ..., p[n_files-1].flac)

    Assumes all flac files have the same length.

    data_path: directory containing the flac files
    n_files: how many FLAC files are in the directory
    (see two_tier.py for a description of the constants)

    returns: (subbatch, reset)
    subbatch.shape: (BATCH_SIZE, SEQ_LEN + OVERLAP)
    reset: True or False
    """

    def round_to(x, y):
        """round x up to the nearest y"""
        return int(numpy.ceil(x / float(y))) * y

    def batch_quantize(data):
        """
        floats in (-1, 1) to ints in [0, Q_LEVELS-1]
        scales normalized across axis 1
        """
        eps = numpy.float64(1e-5)
        data -= data.min(axis=1)[:, None]
        data *= ((Q_LEVELS - eps) / data.max(axis=1)[:, None])
        data += eps/2
        data = data.astype('int32')
        return data%Q_LEVELS

    file_nos = np.random.randint(0,100000,n_files)

    paths = [data_path+'/p{}.flac'.format(i) for i in file_nos]

    SEQ_LEN += 4

    seed = np.zeros((n_files,7)).astype('float32')
    for path in enumerate(paths):
        # batch_seq_len = length of longest sequence in the batch, rounded up to
        # the nearest SEQ_LEN.
        data, fs, enc = scikits.audiolab.flacread(path)
        batch[i, :] = data[:batch_seq_len]

        batch = batch_quantize(batch)

        batch = numpy.concatenate([
            numpy.full((BATCH_SIZE, OVERLAP), Q_ZERO, dtype='int32'),
            batch
        ], axis=1)

        for i in xrange(batch.shape[1] / SEQ_LEN):
            reset = numpy.int32(i==0)
            subbatch = batch[:, i*SEQ_LEN : (i+1)*SEQ_LEN]
#            out1.append(subbatch)
#            out2.append(reset)
            out.append((subbatch,reset))
    return out