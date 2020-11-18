'''
This script use pre-trained model(e.g. FCN)
as the target model. We can query the probability
from it to decide attacks whether efficient or not.
'''
import numpy as np
import torch
import torch.nn as nn


def load_ucr(path, normalize=False):
    data = np.loadtxt(path)
    data[:, 0] -= 1
    # limit label to [0,num_classes-1]
    num_classes = len(np.unique(data[:, 0]))
    for i in range(data.shape[0]):
        if data[i, 0] < 0:
            data[i, 0] = num_classes - 1

    # Normalize some datasets without normalization
    if normalize:
        mean = data[:, 1:].mean(axis=1, keepdims=True)
        std = data[:, 1:].std(axis=1, keepdims=True)
        data[:, 1:] = (data[:, 1:] - mean) / (std + 1e-8)
    return data


def query_one(run_tag, idx, attack_ts, target_class=-1, normalize=False,
              e=1499, verbose=False, cuda=False, model_type='r'):
    device = torch.device("cuda:0" if cuda else "cpu")

    ts = torch.from_numpy(attack_ts).float()
    data_path = 'data/' + run_tag + '/' + run_tag + '_unseen.txt'
    test_data = load_ucr(path=data_path, normalize=normalize)
    test_data = torch.from_numpy(test_data)
    Y = test_data[:, 0]
    n_class = torch.unique(Y).size(0)
    test_one = test_data[idx]

    X = test_one[1:].float()
    y = test_one[0].long()
    y = y.to(device)

    real_label = y

    if target_class != -1:
        y = target_class

    ts = ts.to(device)
    X = X.to(device)
    model_path = 'model_checkpoints/' + run_tag + '/pre_trained.pth'

    model = torch.load(model_path, map_location='cpu')
    with torch.no_grad():

        model.eval()
        softmax = nn.Softmax(dim=-1)

        out = model(X)
        prob_vector = softmax(out)
        prob = prob_vector.view(n_class)[y].item()
        out2 = model(ts)
        prob_vector2 = softmax(out2)
        prob2 = prob_vector2.view(n_class)[y].item()
        if verbose:
            print('Target_Class：', target_class)
            print('Prior Confidence of the %d sample is  %.4f ' % (idx, prob))

    return prob2, prob_vector2, prob, prob_vector, real_label


if __name__ == '__main__':
    query_one('ECG200', 2)
