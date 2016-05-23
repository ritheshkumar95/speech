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
N_FRAMES = 64 # How many 'frames' to include in each truncated BPTT pass
FRAME_SIZE = 4 # How many samples per frame
DIM = 512 # Model dimensionality. 512 is sufficient for model development; 1024 if you want good samples.
N_GRUS = 3 # How many GRUs to stack in the frame-level model
Q_LEVELS = 256 # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
GRAD_CLIP = 1 # Elementwise grad clip threshold

# Dataset
DATA_PATH = '/data/lisatmp3/kumarrit/blizzard'
N_FILES = 100000
BITRATE = 16000

TEST_SET_SIZE = 128 # How many audio files to use for the test set
SEQ_LEN = N_FRAMES * FRAME_SIZE # Total length (# of samples) of each truncated BPTT sequence
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude

print "Model settings:"
all_vars = [(k,v) for (k,v) in locals().items() if (k.isupper() and k != 'T')]
all_vars = sorted(all_vars, key=lambda x: x[0])
for var_name, var_value in all_vars:
    print "\t{}: {}".format(var_name, var_value)

def encoder(input_sequences, h0, reset):
    """
    input_sequences.shape: (batch size, N_FRAMES * FRAME_SIZE)
    h0.shape:              (batch size, N_GRUS, DIM)
    reset.shape:           ()
    output.shape:          (batch size, N_FRAMES * FRAME_SIZE, DIM)
    """
    batch_size = input_sequences.shape[0]
    n_frames = (input_sequences.shape[1]-3)/FRAME_SIZE

    # Rescale frames from ints in [0, Q_LEVELS) to floats in [-2, 2]
    # (a reasonable range to pass as inputs to the RNN)

    emb = lib.ops.Embedding(
        'Embedding',
        Q_LEVELS,
        Q_LEVELS,
        input_sequences,
    ).transpose(0,2,1)    

    #X1 = ((input_sequences.astype(theano.config.floatX)/lib.floatX(Q_LEVELS/2)) - lib.floatX(1))*lib.floatX(2)
    X1 = emb[:,:,None,:] #(128,256,1,259)

    X2 = T.nnet.relu(lib.ops.conv1d('conv1',X1,kernel=4,stride=1,n_filters=512,depth=256,bias=True)) #(128,512,1,256)
    #X3 = T.nnet.relu(lib.ops.conv1d('conv2',X2,kernel=1,stride=1,n_filters=512,depth=512,bias=True)) #(128,512,1,256)
    X4 = lib.ops.pool(X2) #(128,2048,1,64) 

    learned_h0 = lib.param(
        'FrameLevel.h0',
        numpy.zeros((N_GRUS, DIM), dtype=theano.config.floatX)
    )

    learned_h0 = T.alloc(learned_h0, h0.shape[0], N_GRUS, DIM)
    h0 = theano.ifelse.ifelse(reset, learned_h0, h0)

    gru_inp = T.concatenate((X4[:,:,0,:].dimshuffle(0,2,1),emb.transpose(0,2,1)[:,:256,:].reshape((batch_size,n_frames,FRAME_SIZE*Q_LEVELS))),axis=2)
    gru1 = lib.ops.GRU('FrameLevel.GRU1', 3072, DIM, gru_inp, h0=h0[:, 0])
    gru2 = lib.ops.GRU('FrameLevel.GRU2', DIM, DIM, gru1, h0=h0[:, 1])
    gru3 = lib.ops.GRU('FrameLevel.GRU3', DIM, DIM, gru2, h0=h0[:, 2]) ## (128,64,512)

    X9 = lib.ops.Dense(
        'Projection',
        512,
        2048,
        gru3,
        hidden_dim=gru3.shape[1]
        ).reshape((batch_size,4*gru3.shape[1],DIM)).transpose(0,2,1)[:,:,None,:] #(128,64,2048) --> (128,256,512) --> (128,512,256)

    X10 = T.nnet.relu(X9+X2)
    X11 = T.nnet.relu(lib.ops.conv1d('deconv1',X10,kernel=1,stride=1,n_filters=512,depth=512,bias=True)) #(128,512,1,256)
    X12 = T.nnet.relu(lib.ops.conv1d('deconv2',X11,kernel=1,stride=1,n_filters=512,depth=512,bias=True)) #(128,512,1,256)
    X13 = lib.ops.conv1d('deconv3',X12,kernel=1,stride=1,n_filters=256,depth=512,bias=True) #(128,256,1,256)

    last_hidden = T.stack([gru1[:,-1],gru2[:,-1],gru3[:,-1]],axis=1)

    output = X13[:,:,0,:].transpose(0,2,1)


    return (output.reshape((-1,output.shape[2])),last_hidden)


sequences   = T.imatrix('sequences')
h0          = T.ftensor3('h0')
reset       = T.iscalar('reset')
lr          = T.scalar('lr')

input_sequences = sequences[:,:-1]
target_sequences = sequences[:, FRAME_SIZE:]

#prev_samples = sequences[:, :-1]
#prev_samples = prev_samples.reshape((1, BATCH_SIZE, 1, -1))
#prev_samples = T.nnet.neighbours.images2neibs(prev_samples, (1, FRAME_SIZE), neib_step=(1, 1), mode='valid')
#prev_samples = prev_samples.reshape((BATCH_SIZE,SEQ_LEN, FRAME_SIZE))

encoder_outputs, new_h0 = encoder(input_sequences, h0, reset)

#decoder_outputs = decoder(encoder_outputs,prev_samples)

cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(encoder_outputs),
    target_sequences.flatten()
).mean()

cost = cost * lib.floatX(1.44269504089)

params = lib.search(cost, lambda x: hasattr(x, 'param'))

lib.print_params_info(cost, params)

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')

grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

print "Gradients Computed"

updates = lasagne.updates.adam(grads, params, learning_rate = lr)

train_fn = theano.function(
    [sequences, h0, reset,lr],
    [cost, new_h0],
    updates=updates,
    on_unused_input='warn'
)

input_seq = T.imatrix()
hidden_h0 = T.ftensor3()
reset_h = T.scalar()
test_output = encoder(input_seq,hidden_h0,reset_h)
test_fn = theano.function(
    [input_seq,hidden_h0,reset_h],
    [lib.ops.softmax_and_sample(test_output[0]),test_output[1]]
    )

def generate_and_save_samples(tag):

    def write_audio_file(name, data):

        data = data.astype('float32')
        data -= data.min()
        data /= data.max()
        data -= 0.5
        data *= 0.95

        import scipy.io.wavfile
        scipy.io.wavfile.write(name+'.wav',BITRATE,data)

    # Generate 5 sample files, each 5 seconds long
    N_SEQS = 5
    LENGTH = 5*BITRATE

    samples = numpy.zeros((N_SEQS, LENGTH), dtype='int32')
    samples[:, :FRAME_SIZE+3] = Q_ZERO

    h0 = numpy.zeros((N_SEQS, N_GRUS, DIM), dtype='float32')
    h0[0] = h0[1] = h0[2] = lib._params['FrameLevel.h0'].get_value()
    frame_level_outputs = None

    for t in xrange(FRAME_SIZE+3, LENGTH-4,4):

        sample_level_outputs, h0 = test_fn(
            samples[:, t-7:t], 
            h0,
            numpy.int32(0)
        )

        samples[:, t:t+4] = sample_level_outputs.reshape((N_SEQS,4))
        print t

    for i in xrange(N_SEQS):
        write_audio_file("sample_{}_{}".format(tag, i), samples[i])


test = theano.function(
    [sequences, h0, reset,lr],
    [cost, encoder_outputs],
    updates=updates,
    on_unused_input='warn'
)

print "Training!"
total_iters = 0

for epoch in xrange(NB_EPOCH):
    h0 = np.zeros((BATCH_SIZE, N_GRUS, DIM)).astype(theano.config.floatX)
    costs = []
    times = []
    data = dataset.get_data(DATA_PATH, N_FILES, BATCH_SIZE, SEQ_LEN+FRAME_SIZE, 0, Q_LEVELS, Q_ZERO)

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
        #if total_iters%10000==0:
        #    generate_and_save_samples('iterno_%d'%total_iters)
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
