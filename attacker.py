import warnings
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
from query_probability import query_one, load_ucr
warnings.filterwarnings('ignore')


def merge(intervals):
    """
    Merge shapelet interval
    :type intervals: List[List[int]]
    :rtype: List[List[int]]
    :return: the merged shapelet intervals(2d-list)
    """
    if len(intervals) == 0:
        return []

    res = []
    intervals = list(sorted(intervals))
    low = intervals[0][0]
    high = intervals[0][1]

    for i in range(1, len(intervals)):
        if high >= intervals[i][0]:
            if high < intervals[i][1]:
                high = intervals[i][1]
        else:
            res.append([low, high])
            low = intervals[i][0]
            high = intervals[i][1]

    res.append([low, high])

    return res


def get_interval(run_tag, topk):
    '''
    :param topk: the k shapelets
    :param run_tag: e.g. ECG200
    :return: shapelet interval  after merging
    '''
    shaplet_pos = np.loadtxt('shapelet_pos/' + run_tag + '_shapelet_pos.txt', usecols=(2, 3))
    shaplet_pos = shaplet_pos[:topk]
    shaplet_pos = shaplet_pos.tolist()

    return merge(shaplet_pos)


def get_magnitude(run_tag, factor, normalize):
    '''
    :param run_tag:
    :param factor:
    :return: Perturbed Magnitude
    '''
    data = load_ucr('data/' + run_tag + '/' + run_tag + '_unseen.txt', normalize=normalize)
    X = data[:, 1:]

    max_magnitude = X.max(1)
    min_magnitude = X.min(1)
    mean_magnitude = np.mean(max_magnitude - min_magnitude)

    perturbed_mag = mean_magnitude * factor
    print('Perturbed Magnitude:', perturbed_mag)

    return perturbed_mag


class Attacker:
    def __init__(self, run_tag, top_k, model_type, cuda, normalize, e):
        self.run_tag = run_tag
        self.top_k = top_k
        self.model_type = model_type
        self.cuda = cuda
        self.intervals = get_interval(self.run_tag, self.top_k)
        self.normalize = normalize
        self.e = e

    def perturb_ts(self, perturbations, ts):
        '''
        :param perturbations:formalized as a tuple（x,e),x(int) is the x-coordinate，e(float) is the epsilon,e.g.,(2,0.01)
        :param ts: time series
        :return: perturbed ts
        '''
        # first we copy a ts
        ts_tmp = np.copy(ts)
        coordinate = 0
        for interval in self.intervals:
            for i in range(int(interval[0]), int(interval[1])):
                ts_tmp[i] += perturbations[coordinate]
                coordinate += 1
        return ts_tmp

    def plot_per(self, perturbations, ts, target_class, sample_idx, prior_probs, attack_probs, factor):

        # Obtain the perturbed ts
        ts_tmp = np.copy(ts)
        ts_perturbed = self.perturb_ts(perturbations=perturbations, ts=ts)
        # Start to plot
        plt.figure(figsize=(6, 4))
        plt.plot(ts_tmp, color='b', label='Original %.2f' % prior_probs)
        plt.plot(ts_perturbed, color='r', label='Perturbed %.2f' % attack_probs)
        plt.xlabel('Time', fontsize=12)

        if target_class == -1:
            plt.title('Untargeted: Sample %d, eps_factor=%.3f' %
                      (sample_idx, factor), fontsize=14)
        else:
            plt.title('Targeted(%d): Sample %d, eps_factor=%.3f' %
                      (target_class, sample_idx, factor), fontsize=14)

        plt.legend(loc='upper right', fontsize=8)
        plt.savefig('result_' + str(factor) + '_' + str(self.top_k) + '_' + str(self.model_type) + '/'
                    + self.run_tag + '/figures/' + self.run_tag + '_' + str(sample_idx) + '.png')
        # plt.show()

    def fitness(self, perturbations, ts, sample_idx, queries, target_class=-1):

        queries[0] += 1
        ts_perturbed = self.perturb_ts(perturbations, ts)
        prob, _, _, _, _ = query_one(run_tag=self.run_tag, idx=sample_idx, attack_ts=ts_perturbed,
                                     target_class=target_class, normalize=self.normalize,
                                     cuda=self.cuda, model_type=self.model_type, e=self.e)

        if target_class != -1:
            prob = 1 - prob

        return prob  # The fitness function is to minimize the fitness value

    def attack_success(self, perturbations, ts, sample_idx, iterations, target_class=-1, verbose=True):

        iterations[0] += 1
        print('The %d iteration' % iterations[0])
        ts_perturbed = self.perturb_ts(perturbations, ts)
        # Obtain the perturbed probability vector and the prior probability vector
        prob, prob_vector, prior_prob, prior_prob_vec, real_label = query_one(self.run_tag, idx=sample_idx,
                                                                              attack_ts=ts_perturbed,
                                                                              target_class=target_class,
                                                                              normalize=self.normalize,
                                                                              verbose=verbose, cuda=self.cuda,
                                                                              model_type=self.model_type,
                                                                              e=self.e)

        predict_class = torch.argmax(prob_vector)
        prior_class = torch.argmax(prior_prob_vec)

        # Conditions for early termination(empirical-based estimation), leading to save the attacking time
        # But it may judge incorrectly that this may decrease the success rate of the attack.
        if (iterations[0] > 5 and prob > 0.99) or \
                (iterations[0] > 20 and prob > 0.9):
            print('The %d sample sample is not expected to successfully attack.' % sample_idx)
            return True

        if prior_class != real_label:
            print('The %d sample cannot be classified correctly, no need to attack' % sample_idx)
            return True

        if prior_class == target_class:
            print(
                'The true label of %d sample equals to target label, no need to attack' % sample_idx)
            return True

        if verbose:
            print('The Confidence of current iteration: %.4f' % prob)
            print('########################################################')

        # The criterion of attacking successfully:
        # Untargeted attack: predicted label is not equal to the original label.
        # Targeted attack: predicted label is equal to the target label.
        if ((target_class == -1 and predict_class != prior_class) or
                (target_class != -1 and predict_class == target_class)):
            print('##################### Attack Successfully! ##########################')

            return True

    def attack(self, sample_idx, target_class=-1, factor=0.04,
               max_iteration=60, popsize=200, verbose=True):

        test = load_ucr('data/' + self.run_tag + '/' + self.run_tag + '_unseen.txt'
                        , normalize=self.normalize)
        ori_ts = test[sample_idx][1:]

        attacked_probs, attacked_vec, prior_probs, prior_vec, real_label = query_one(self.run_tag, idx=sample_idx,
                                                                                     attack_ts=ori_ts,
                                                                                     target_class=target_class,
                                                                                     normalize=self.normalize,
                                                                                     verbose=False,
                                                                                     cuda=self.cuda, e=self.e,
                                                                                     model_type=self.model_type)
        prior_class = torch.argmax(prior_vec)
        if prior_class != real_label:
            print('The %d sample cannot be classified correctly, no need to attack' % sample_idx)
            return ori_ts, [prior_probs, attacked_probs, 0, 0, 0, 0, 0, 'WrongSample']

        steps_count = 0  # count the number of coordinates

        # Get the maximum perturbed magnitude
        perturbed_magnitude = get_magnitude(self.run_tag, factor, normalize=self.normalize)

        bounds = []

        for interval in self.intervals:
            steps_count += int(interval[1]) - int(interval[0])
            for i in range(int(interval[0]), int(interval[1])):
                bounds.append((-1 * perturbed_magnitude, perturbed_magnitude))
        print('The length of shapelet interval', steps_count)
        popmul = max(1, popsize // len(bounds))
        # Record of the number of iterations
        iterations = [0]
        queries = [0]

        def fitness_fn(perturbations):

            return self.fitness(perturbations=perturbations, ts=ori_ts, queries=queries,
                                sample_idx=sample_idx, target_class=target_class)

        def callback_fn(x, convergence):

            return self.attack_success(perturbations=x, ts=ori_ts,
                                       sample_idx=sample_idx,
                                       iterations=iterations,
                                       target_class=target_class,
                                       verbose=verbose)

        attack_result = differential_evolution(func=fitness_fn, bounds=bounds
                                               , maxiter=max_iteration, popsize=popmul
                                               , recombination=0.7, callback=callback_fn,
                                               atol=-1, polish=False)

        attack_ts = self.perturb_ts(attack_result.x, ori_ts)

        mse = mean_squared_error(ori_ts, attack_ts)

        attacked_probs, attacked_vec, prior_probs, prior_vec, real_label = query_one(self.run_tag, idx=sample_idx,
                                                                                     attack_ts=attack_ts,
                                                                                     target_class=target_class,
                                                                                     normalize=self.normalize,
                                                                                     verbose=False,
                                                                                     cuda=self.cuda, e=self.e,
                                                                                     model_type=self.model_type)

        predicted_class = torch.argmax(attacked_vec)
        prior_class = torch.argmax(prior_vec)

        if prior_class != real_label:
            success = 'WrongSample'

        elif prior_class == target_class:
            success = 'NoNeedAttack'

        else:
            if (predicted_class.item() != prior_class.item() and target_class == -1) \
                    or (predicted_class.item() == target_class and target_class != -1):
                success = 'Success'
            else:
                success = 'Fail'

        if success == 'Success':
            self.plot_per(perturbations=attack_result.x, ts=ori_ts, target_class=target_class,
                          sample_idx=sample_idx, prior_probs=prior_probs, attack_probs=attacked_probs, factor=factor)

        return attack_ts, [prior_probs, attacked_probs, prior_class.item(),
                           predicted_class.item(), queries[0], mse, iterations[0], success]


if __name__ == '__main__':
    idx = [1]
    attacker = Attacker(run_tag='ECG200', top_k=3, model_type='f', e=1499,
                        cuda=False, normalize=False)
    for idx in idx:
        attack_ts, info = attacker.attack(sample_idx=idx, target_class=-1, factor=0.01,
                                          max_iteration=200, popsize=1, verbose=True)

        if info[-1] == 'Success':
            file = open('attack_time_series.txt', 'w+')
            file.write('%d %d ' % (idx, info[3]))  # save the sample index and the perturbed label
            for i in attack_ts:
                file.write('%.4f ' % i)
            file.write('\n')
            file.close()

        print(info)
        file = open('info.txt', 'w+')
        file.write('%d ' % idx)
        for i in info:
            if isinstance(i, int):
                file.write('%d ' % i)
            elif isinstance(i, float):
                file.write('%.4f ' % i)
            else:
                file.write(i + ' ')

        file.write('\n')
        file.close()
