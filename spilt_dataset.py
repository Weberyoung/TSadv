import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(x_test, y_test, test_fraction=0.5):
    """
    Accepts an inputs dataset, and selects a portion of the test set to be
    the new train set for the adversarial model.

    Tries to extract such that number of samples in the new train set is
    same as the number of samples in the original train set.

    Uses class wise splitting to maintain counts from the test set.

    Args:
        X_test: numpy array
        y_test: numpy array

    Returns:
        (X_train, y_train), (X_test, y_test) with reduced number of samples
    """
    np.random.seed(0)
    # num_classes = y_test.shape[-1]
    # print(num_classes)
    # y_test = checked_argmax(y_test)

    test_labels, test_counts = np.unique(y_test, return_counts=True)

    X_train, Y_train = [], []
    X_test, Y_test = [], []
    # Split the test set into adversarial train and adversarial test splits
    for label, max_cnt in zip(test_labels, test_counts):
        samples = x_test[y_test == label]
        if len(samples) == 1:
            X_train.append(samples)
            Y_train.append([label] * 1)
            X_test.append(samples)
            Y_test.append([label] * 1)
            continue
        train_samples, test_samples = train_test_split(samples, test_size=test_fraction, random_state=0)

        train_cnt = len(train_samples)
        max_cnt = train_cnt
        print()
        X_train.append(train_samples[:max_cnt])
        Y_train.append([label] * max_cnt)
        X_test.append(test_samples)
        Y_test.append([label] * len(test_samples))
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    Y_train = np.concatenate(Y_train)
    Y_test = np.concatenate(Y_test)
    # print("\nSplitting test set into new train and test set !")
    print("X train = ", X_train.shape, "Y train : ", Y_train.shape)
    print("X test = ", X_test.shape, "Y test : ", Y_test.shape)
    train_data = np.insert(X_train, 0, values=Y_train, axis=1)
    test_data = np.insert(X_test, 0, values=Y_test, axis=1)

    return train_data, test_data


if __name__ == '__main__':
    name = ['Beef',  # ECG+Sensor:24 spectrum: 7
            'Car', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee',
            'Earthquakes', 'ECG5000', 'ECG200', 'ECGFiveDays',
            'FordA', 'FordB',
            'Ham',
            'InsectWingbeatSound', 'ItalyPowerDemand',
            'Lightning2', 'Lightning7',
            'Meat', 'MoteStrain',
            'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2',
            'OliveOil',
            'Plane', 'Phoneme',
            'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry',
            'Trace', 'TwoLeadECG',
            'Wafer', 'Wine',
            'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ',  # ECG+Sensor+HRM:18
            'FreezerRegularTrain', 'FreezerSmallTrain',
            'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP',
            'ShakeGestureWiimoteZ',
            'Fungi',
            'GesturePebbleZ1', 'GesturePebbleZ2',
            'DodgerLoopDay', 'DodgerLoopWeekend', 'DodgerLoopGame',
            'EOGHorizontalSignal', 'EOGVerticalSignal']
    for n in name:
        print('\nSpiltting test set on %s....' % n)
        testset = np.loadtxt('data/' + n + '/' + n + '_TEST.txt')
        x_test = testset[:, 1:]
        y_test = testset[:, 0]
        train, test = split_dataset(x_test, y_test)
        np.savetxt('data/%s/%s_eval.txt' % (n, n), train)
        np.savetxt('data/%s/%s_unseen.txt' % (n, n), test)
