import lib
import numpy as np
import numpy
import theano
import theano.tensor as T
theano.config.floatX='float32'
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams(seed=234)

def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out

def glorot_uniform(shape,init='glorot'):
    def uniform(shape, scale=0.05, name=None):
        return np.random.uniform(low=-scale, high=scale, size=shape)
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    if init=='he':
        s = np.sqrt(6./fan_in)
        return uniform(shape,s)
    else:
        return uniform(shape, s)

def init_weights(fan_in,fan_out,init='he'):

    def uniform(stdev, size):
        """uniform distribution with the given stdev and size"""
        return numpy.random.uniform(
            low=-stdev * numpy.sqrt(3),
            high=stdev * numpy.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    if init == 'lecun' or (init == None and fan_in != fan_out):
        weight_values = uniform(numpy.sqrt(1. / fan_in), (fan_in, fan_out))

    elif init == 'he':
        weight_values = uniform(numpy.sqrt(2. / fan_in), (fan_in, fan_out))
    
    elif init == 'orthogonal' or (init == None and fan_in == fan_out):
        # From lasagne
        def sample(shape):
            if len(shape) < 2:
                raise RuntimeError("Only shapes of length 2 or more are "
                                   "supported.")
            flat_shape = (shape[0], numpy.prod(shape[1:]))
            # TODO: why normal and not uniform?
            a = numpy.random.normal(0.0, 1.0, flat_shape)
            u, _, v = numpy.linalg.svd(a, full_matrices=False)
            # pick the one with the correct shape
            q = u if u.shape == flat_shape else v
            q = q.reshape(shape)
            return q.astype(theano.config.floatX)
        weight_values = sample((fan_in, fan_out))
    return weight_values

def Dense(name, input_dim, output_dim, inputs, bias=True, init=None, weightnorm=True,hidden_dim=None):

    weight_values = init_weights(input_dim,output_dim,init)

    weight = lib.param(
        name + '.W',
        weight_values
    )

    batch_size = None
    if inputs.ndim==3:
        batch_size = inputs.shape[0]
        inputs = inputs.reshape((-1,input_dim))

    if weightnorm:
        norm_values = numpy.linalg.norm(weight_values, axis=0)
        norms = lib.param(
            name + '.g',
            norm_values
        )

        normed_weight = weight * (norms / weight.norm(2, axis=0)).dimshuffle('x', 0)
        result = T.dot(inputs, normed_weight)

    else:        
        result = T.dot(inputs, weight)

    if bias:
        b = lib.param(
            name + '.b',
            numpy.zeros((output_dim,), dtype=theano.config.floatX)
        )
        result += b

    result.name = name+".output"
    if batch_size!=None:
        return result.reshape((batch_size,hidden_dim,output_dim))
    else:
        return result

def Embedding(name, n_symbols, output_dim, indices):
    vectors = lib.param(
        name,
        numpy.random.randn(
            n_symbols, 
            output_dim
        ).astype(theano.config.floatX)
    )

    output_shape = tuple(list(indices.shape) + [output_dim])

    return vectors[indices.flatten()].reshape(output_shape)

def softmax_and_sample(logits):
    old_shape = logits.shape
    flattened_logits = logits.reshape((-1, logits.shape[logits.ndim-1]))
    samples = T.cast(
        srng.multinomial(pvals=T.nnet.softmax(flattened_logits)),
        theano.config.floatX
    ).reshape(old_shape)
    return T.argmax(samples, axis=samples.ndim-1)

def GRUStep(name, input_dim, hidden_dim, x_t, h_tm1):
    processed_input = lib.ops.Dense(
        name+'.Input',
        input_dim,
        3 * hidden_dim,
        x_t
    )

    gates = T.nnet.sigmoid(
        lib.ops.Dense(
            name+'.Recurrent_Gates',
            hidden_dim,
            2 * hidden_dim,
            h_tm1,
            bias=False
        ) + processed_input[:, :2*hidden_dim]
    )

    update = gates[:, :hidden_dim]
    reset  = gates[:, hidden_dim:]

    scaled_hidden = reset * h_tm1

    candidate = T.tanh(
        lib.ops.Dense(
            name+'.Recurrent_Candidate', 
            hidden_dim, 
            hidden_dim, 
            scaled_hidden,
            bias=False,
            init='orthogonal'
        ) + processed_input[:, 2*hidden_dim:]
    )

    one = lib.floatX(1.0)
    return (update * candidate) + ((one - update) * h_tm1)

def GRU(name, input_dim, hidden_dim, inputs, h0=None):
    #inputs.shape = (batch_size,N_FRAMES,FRAME_SIZE)
    inputs = inputs.transpose(1,0,2)

    def step(x_t, h_tm1):
        return GRUStep(
            name+'.Step', 
            input_dim, 
            hidden_dim, 
            x_t, 
            h_tm1
        )

    outputs, _ = theano.scan(
        step,
        sequences=[inputs],
        outputs_info=[h0],
    )

    out = outputs.dimshuffle(1,0,2)
    out.name = name+'.output'
    return out

def recurrent_fn(x_t, h_tm1,hidden_dim,W1,b1,W2,b2):
    A1 = T.nnet.sigmoid(T.dot(T.concatenate((x_t,h_tm1),axis=1),W1) + b1)

    z = A1[:,:hidden_dim]

    r = A1[:,hidden_dim:]

    scaled_hidden = r*h_tm1

    h = T.tanh(T.dot(T.concatenate((scaled_hidden,x_t),axis=1),W2)+b2)

    one = lib.floatX(1.0)
    return ((z * h) + ((one - z) * h_tm1)).astype('float32')

def myGRU(name, input_dim, hidden_dim, inputs, h0=None):
    #inputs.shape = (batch_size,N_FRAMES,FRAME_SIZE)
    inputs = inputs.transpose(1,0,2)

    weight_values = init_weights(input_dim+hidden_dim,2*hidden_dim)
    W1 = lib.param(
        name+'.Gates.W',
        weight_values
    )

    norm_values = numpy.linalg.norm(weight_values, axis=0)
    norms = lib.param(
        name + 'Gates.W.g',
        norm_values
    )
    n_W1 = W1 * (norms / W1.norm(2, axis=0)).dimshuffle('x', 0)

    b1 = lib.param(
        name+'.Gates.b',
        np.ones(2*hidden_dim).astype(theano.config.floatX)
        )

    weight_values = init_weights(input_dim+hidden_dim,hidden_dim)
    W2 = lib.param(
        name+'.Candidate.W',
        weight_values
    )

    norm_values = numpy.linalg.norm(weight_values, axis=0)
    norms = lib.param(
        name + 'Candidate.W.g',
        norm_values
    )
    n_W2 = W2 * (norms / W2.norm(2, axis=0)).dimshuffle('x', 0)

    b2 = lib.param(
        name+'.Candidate.b',
        np.zeros(hidden_dim).astype(theano.config.floatX)
        )

    outputs, _ = theano.scan(
        recurrent_fn,
        sequences=[inputs],
        outputs_info=[h0],
        non_sequences=[hidden_dim,n_W1,b1,n_W2,b2]
    )

    out = outputs.dimshuffle(1,0,2)
    out.name = name+'.output'
    return out

def recurrent_fn2(x_t, h1_tm1, h2_tm1, h3_tm1,hidden_dim,W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6):
    A1 = T.nnet.sigmoid(T.dot(T.concatenate((x_t,h1_tm1),axis=1),W1) + b1)

    z1 = A1[:,:hidden_dim]

    r1 = A1[:,hidden_dim:]

    scaled_hidden1 = r1*h1_tm1

    h1 = T.tanh(T.dot(T.concatenate((scaled_hidden1,x_t),axis=1),W2)+b2)

    one = lib.floatX(1.0)

    h1_t =  (z1 * h1) + ((one - z1) * h1_tm1)

######################################################################

    A2 = T.nnet.sigmoid(T.dot(T.concatenate((h1_t,h2_tm1),axis=1),W3) + b3)

    z2 = A2[:,:hidden_dim]

    r2 = A2[:,hidden_dim:]

    scaled_hidden2 = r2*h2_tm1

    h2 = T.tanh(T.dot(T.concatenate((scaled_hidden2,h1_t),axis=1),W4)+b4)

    h2_t =  (z2 * h2) + ((one - z2) * h2_tm1)

########################################################################

    A3 = T.nnet.sigmoid(T.dot(T.concatenate((h2_t,h3_tm1),axis=1),W5) + b5)

    z3 = A3[:,:hidden_dim]

    r3 = A3[:,hidden_dim:]

    scaled_hidden3 = r3*h3_tm1

    h3 = T.tanh(T.dot(T.concatenate((scaled_hidden3,h2_t),axis=1),W6)+b6)

    h3_t =  (z3 * h3) + ((one - z3) * h3_tm1)

    return h1_t,h2_t,h3_t

def myGRU2(name, input_dim, hidden_dim, inputs, h0=None):
    #inputs.shape = (batch_size,N_FRAMES,FRAME_SIZE)
    inputs = inputs.transpose(1,0,2)

    W1 = lib.param(
        name+'.Layer1.Gates.W',
        init_weights(input_dim+hidden_dim,2*hidden_dim)
    )

    b1 = lib.param(
        name+'.Layer1.Gates.b',
        np.ones(2*hidden_dim).astype(theano.config.floatX)
        )

    W2 = lib.param(
        name+'.Layer1.Candidate.W',
        init_weights(input_dim+hidden_dim,hidden_dim)
    )

    b2 = lib.param(
        name+'.Layer1.Candidate.b',
        np.zeros(hidden_dim).astype(theano.config.floatX)
        )

    W3 = lib.param(
        name+'.Layer2.Gates.W',
        init_weights(hidden_dim+hidden_dim,2*hidden_dim)
    )

    b3 = lib.param(
        name+'.Layer2.Gates.b',
        np.ones(2*hidden_dim).astype(theano.config.floatX)
        )

    W4 = lib.param(
        name+'.Layer2.Candidate.W',
        init_weights(hidden_dim+hidden_dim,hidden_dim)
    )

    b4 = lib.param(
        name+'.Layer2.Candidate.b',
        np.zeros(hidden_dim).astype(theano.config.floatX)
        )

    W5 = lib.param(
        name+'.Layer3.Gates.W',
        init_weights(hidden_dim+hidden_dim,2*hidden_dim)
    )

    b5 = lib.param(
        name+'.Layer3.Gates.b',
        np.ones(2*hidden_dim).astype(theano.config.floatX)
        )

    W6 = lib.param(
        name+'.Layer3.Candidate.W',
        init_weights(hidden_dim+hidden_dim,hidden_dim)
    )

    b6 = lib.param(
        name+'.Layer3.Candidate.b',
        np.zeros(hidden_dim).astype(theano.config.floatX)
        )

    [out1,out2,out3], _ = theano.scan(
        recurrent_fn2,
        sequences=[inputs],
        outputs_info=[h0[:,0,:],h0[:,1,:],h0[:,2,:]],
        non_sequences=[hidden_dim,W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6]
    )

    out1 = out1.dimshuffle(1,0,2)
    out2 = out2.dimshuffle(1,0,2)
    out3 = out3.dimshuffle(1,0,2)

    out1.name = name+'.output'
    out2.name = name+'.output'
    out3.name = name+'.output'

    return out1,out2,out3

def conv1d(name,input,kernel,stride,n_filters,depth,bias=False):
    W = lib.param(
        name+'.W',
        glorot_uniform((n_filters,depth,1,kernel)).astype('float32')
        )

    if bias:
        b = lib.param(
            name + '.b',
            np.zeros(n_filters).astype('float32')
            )

    return T.nnet.conv2d(input,W,filter_flip=False,subsample=(1,stride)) + b[None,:,None,None]

def pool(input):
    x = input[:,:,0,:]
    return x.reshape((-1,x.shape[1]*4,x.shape[2]/4))[:,:,None,:]

def subsample(input):
    x = input[:,:,0,:]
    idx = T.arange(0,x.shape[2],4)
    return x[:,:,idx][:,:,None,:]

def upsample(input):
    x = input[:,:,0,:]
    return x.reshape((-1,x.shape[1]/4,x.shape[2]*4))[:,:,None,:]



