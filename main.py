import argparse
import numpy as np
import os
import time
from attacker import Attacker

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='it is unnecessary')
    parser.add_argument('--target_class', type=int, default=-1, help='-1:untargeted')
    parser.add_argument('--popsize', type=int, default=1, help='the popsuze of DE')
    parser.add_argument('--magnitude_factor', type=float, default=0.05, help='the value of beta')
    parser.add_argument('--maxitr', type=int, default=50, help='max iterations of DE')
    parser.add_argument('--run_tag', default='', help='the name of dataset e.g.ECG200')
    parser.add_argument('--model', default='f', help='the model type(ResNet,FCN),f:FCN r:Resnet')
    parser.add_argument('--topk', type=int, default=3, help='employ the top k shapelets, maxima value is 5')
    parser.add_argument('--normalize', action='store_true', help='it is unnecessary in our project, we have '
                                                                 'normalized the data')
    parser.add_argument('--e', type=int, default=1499, help='epochs of model')

    opt = parser.parse_args()
    print(opt)

    os.makedirs('result_%s_%d_%s/%s/figures' % (str(opt.magnitude_factor), opt.topk,
                                                opt.model, opt.run_tag), exist_ok=True)
    data_path = 'data/' + opt.run_tag + '/' + opt.run_tag + '_unseen.txt'
    test_data = np.loadtxt(data_path)

    size = test_data.shape[0]
    idx_array = np.arange(size)
    attacker = Attacker(run_tag=opt.run_tag, top_k=opt.topk, e=opt.e,
                        model_type=opt.model, cuda=opt.cuda, normalize=opt.normalize)
    # record of the running time
    start_time = time.time()
    # count the number of the successful instances, mse,iterations,queries
    success_cnt = 0
    right_cnt = 0
    total_mse = 0
    total_iterations = 0
    total_quries = 0
    for idx in idx_array:
        print('###Start %s : generating adversarial example of the %d sample ###' % (opt.run_tag, idx))
        attack_ts, info = attacker.attack(sample_idx=idx, target_class=opt.target_class,
                                          factor=opt.magnitude_factor, max_iteration=opt.maxitr,
                                          popsize=opt.popsize)

        # only save the successful adversarial example
        if info[-1] == 'Success':
            success_cnt = success_cnt + 1
            total_iterations += info[-2]
            total_mse += info[-3]
            total_quries += info[-4]

            file = open('result_' + str(opt.magnitude_factor) + '_' + str(opt.topk) + '_' + opt.model
                        + '/' + opt.run_tag + '/attack_time_series.txt', 'a+')
            file.write('%d %d ' % (idx, info[3]))
            for i in attack_ts:
                file.write('%.4f ' % i)
            file.write('\n')
            file.close()

        if info[-1] != 'WrongSample':
            right_cnt += 1

        # Save the returned information, whether the attack was successful or not
        file = open('result_' + str(opt.magnitude_factor) + '_' + str(opt.topk) + '_' + opt.model
                    + '/' + opt.run_tag + '/information.txt', 'a+')

        file.write('%d ' % idx)
        for i in info:
            if isinstance(i, int):
                file.write('%d ' % i)
            elif isinstance(i, float):
                file.write('%.4f ' % i)
            else:
                file.write(str(i) + ' ')
        file.write('\n')
        file.close()

    endtime = time.time()
    total = endtime - start_time
    # print useful information
    print('Running time: %.4f ' % total)
    print('Correctly-classified samples: %d' % right_cnt)
    print('Successful samples: %d' % success_cnt)
    print('Success rate：%.2f%%' % (success_cnt / right_cnt * 100))
    print('Misclassification rate：%.2f%%' % (success_cnt / size * 100))
    print('ANI: %.2f' % (total_iterations / success_cnt))
    print('MSE: %.4f' % (total_mse / success_cnt))
    print('Mean queries：%.2f\n' % (total_quries / success_cnt))

    # save the useful information
    file = open('result_' + str(opt.magnitude_factor) + '_' + str(opt.topk) + '_' + opt.model
                + '/' + opt.run_tag + '/information.txt', 'a+')
    file.write('Running time:%.4f\n' % total)
    file.write('Correctly-classified samples: %d' % right_cnt)
    file.write('Successful samples:%d\n' % success_cnt)
    file.write('Success rate：%.2f%%' % (success_cnt / right_cnt * 100))
    file.write('Misclassification rate：%.2f%%\n' % (success_cnt / size * 100))
    file.write('ANI:%.2f\n' % (total_iterations / success_cnt))
    file.write('MSE:%.4f\n' % (total_mse / success_cnt))
    file.write('Mean queries：%.2f\n' % (total_quries / success_cnt))

    file.close()
