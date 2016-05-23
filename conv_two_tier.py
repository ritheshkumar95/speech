import numpy as np
import numpy
numpy.random.seed(123)
import random
random.seed(123)

import dataset

import theano
import theano.tensor as T
theano.config.floatX='float32'
from theano.tensor.nnet import neighbours
import theano.ifelse
import lib
import lasagne
import scipy.io.wavfile
import time

# Hyperparams
NB_EPOCH=10
BATCH_SIZE = 128
N_FRAMES = 256 # How many 'frames' to include in each truncated BPTT pass
FRAME_SIZE = 4 # How many samples per frame
DIM = 512 # Model dimensionality. 512 is sufficient for model development; 1024 if you want good samples.
N_GRUS = 3 # How many GRUs to stack in the frame-level model
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
GRAD_CLIP = 1 # Elementwise grad clip threshold

# Dataset
DATA_PATH = '/data/lisatmp3/kumarrit/blizzard'
N_FILES = 1000
BITRATE = 16000

TEST_SET_SIZE = 128 # How many audio files to use for the test set
SEQ_LEN = N_FRAMES * FRAME_SIZE # Total length (# of samples) of each truncated BPTT sequence
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude

print "Model settings:"
all_vars = [(k,v) for (k,v) in locals().items() if (k.isupper() and k != 'T')]
all_vars = sorted(all_vars, key=lambda x: x[0])
for var_name, var_value in all_vars:
    print "\t{}: {}".format(var_name, var_value)

def encoder_decoder(input_sequences, h0, reset):
    """
    input_sequences.shape: (batch size, N_FRAMES * FRAME_SIZE)
    h0.shape:              (batch size, N_GRUS, DIM)
    reset.shape:           ()
    output.shape:          (batch size, N_FRAMES * FRAME_SIZE, DIM)
    """
    batch_size = input_sequences.shape[0]
    n_frames = input_sequences.shape[1]/FRAME_SIZE

    # Rescale frames from ints in [0, Q_LEVELS) to floats in [-2, 2]
    # (a reasonable range to pass as inputs to the RNN)
    
    X1 = ((input_sequences.astype(theano.config.floatX)/lib.floatX(Q_LEVELS/2)) - lib.floatX(1))*lib.floatX(2)
    X1 = X1[:,None,None,:]

    X2 = T.nnet.relu(lib.ops.conv1d('conv1',X1,kernel=4,stride=1,n_filters=128,depth=1))
    X3 = T.nnet.relu(lib.ops.conv1d('conv2',X2,kernel=6,stride=1,n_filters=64,depth=128))
    X4 = lib.ops.pool(X3) #(batch_size,256,1,62)

    X5 = T.nnet.relu(lib.ops.conv1d('conv3',X4,kernel=4,stride=1,n_filters=128,depth=256))
    X6 = T.nnet.relu(lib.ops.conv1d('conv4',X5,kernel=4,stride=1,n_filters=128,depth=128))

    X7 = lib.ops.pool(X6)

    learned_h0 = lib.param(
        'FrameLevel.h0',
        numpy.zeros((N_GRUS, DIM), dtype=theano.config.floatX)
    )

    learned_h0 = T.alloc(learned_h0, h0.shape[0], N_GRUS, DIM)
    h0 = theano.ifelse.ifelse(reset, learned_h0, h0)

    gru_inp = X7[:,:,0,:].dimshuffle(0,2,1)
    gru1 = lib.ops.myGRU('FrameLevel.GRU1', DIM, DIM, gru_inp, h0=h0[:, 0])
    gru2 = lib.ops.myGRU('FrameLevel.GRU2', DIM, DIM, gru1, h0=h0[:, 1])
    gru3 = lib.ops.myGRU('FrameLevel.GRU3', DIM, DIM, gru2, h0=h0[:, 2])

    X8 = gru3.transpose(0,2,1)[:,:,None,:]
    X9 = lib.ops.upsample(X8)

    #Skip connectoin
    X10 = X9 + lib.ops.Dense(
        'SkipConnection1',
        128,
        128,
        X6[:,:,0,:].transpose(0,2,1),
        init='he',
        hidden_dim=X6.shape[3]
        ).transpose(0,2,1)[:,:,None,:]

    X11 = T.nnet.relu(lib.ops.conv1d('deconv1',X10,kernel=4,stride=1,n_filters=128,depth=128))
    X12 = T.nnet.relu(lib.ops.conv1d('deconv2',X11,kernel=4,stride=1,n_filters=128,depth=128))
    X13 = lib.ops.upsample(X12)

    #x3.shape (212,64)
    #SkipConnection 2
    X14 = X13 + lib.ops.Dense(
        'SkipConnection2',
        64,
        32,
        X3[:,:,0,:].transpose(0,2,1)[:,:968],
        hidden_dim=968
        ).transpose(0,2,1)[:,:,None,:]

    X15 = T.nnet.relu(lib.ops.conv1d('deconv3',X14,kernel=4,stride=1,n_filters=128,depth=32))
    X16 = T.nnet.relu(lib.ops.conv1d('deconv4',X15,kernel=4,stride=1,n_filters=256,depth=128))

    ##194

    output = X16[:,:,0,:].transpose(0,2,1)
    last_hidden = T.stack([gru1[:,-1],gru2[:,-1],gru3[:,-1]],axis=1)

    return (output.reshape((-1,output.shape[2])), last_hidden)


sequences   = T.imatrix('sequences')
h0          = T.ftensor3('h0')
reset       = T.iscalar('reset')
lr          = T.scalar('lr')

input_sequences = sequences[:, :-FRAME_SIZE]
target_sequences = sequences[:, 62:-FRAME_SIZE]

decoder_outputs, new_h0 = encoder_decoder(input_sequences, h0, reset)

cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(decoder_outputs),
    target_sequences.flatten()
).mean()

cost = cost * lib.floatX(1.44269504089)

params = lib.search(cost, lambda x: hasattr(x, 'param'))

lib.print_params_info(cost, params)

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')

print "Gradients Computed"

updates = lasagne.updates.adam(grads, params, learning_rate = lr)

train_fn = theano.function(
    [sequences, h0, reset,lr],
    [cost, new_h0],
    updates=updates,
    on_unused_input='warn'
)

test = theano.function(
    [sequences, h0, reset,lr],
    [cost, decoder_outputs],
    updates=updates,
    on_unused_input='warn'
)

print "Training!"
total_iters = 0

for epoch in xrange(NB_EPOCH):
    h0 = np.zeros((BATCH_SIZE, N_GRUS, DIM)).astype(theano.config.floatX)
    costs = []
    times = []
    data = dataset.get_data(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN, FRAME_SIZE, Q_LEVELS, Q_ZERO)

    for seqs, reset in data:
        start_time = time.time()
        cost, h0 = train_fn(seqs, h0, reset, 0.001)
        total_time = time.time() - start_time
        times.append(total_time)
        total_iters += 1
        print "Batch ",total_iters
        costs.append(cost)
        print "\tBatch Cost: ",cost
        print "\t Mean Cost: ",np.mean(costs)
        print "\tTime: ",np.mean(times)
        if total_iters%10000==0:
            generate_and_save_samples('iterno_%d'%total_iters)
    break        
        # print "epoch:{}\ttotal iters:{}\ttrain cost:{}\ttotal time:{}\ttime per iter:{}".format(
        #     epoch,
        #     total_iters,
        #     numpy.mean(costs),
        #     total_time,
        #     total_time / total_iters
        # )
        # tag = "iters{}_time{}".format(total_iters, total_time)
        # generate_and_save_samples(tag)
        # lib.save_params('params_{}.pkl'.format(tag))

        # costs = []
        # last_print_time += PRINT_TIME
        # last_print_iters += PRINT_ITERS
