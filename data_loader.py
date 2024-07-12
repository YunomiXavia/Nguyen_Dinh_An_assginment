import scipy.io as sio

def load_data():
    pos_samples = sio.loadmat('datasets/possamples.mat')['possamples']
    neg_samples = sio.loadmat('datasets/negsamples.mat')['negsamples']
    return pos_samples, neg_samples
