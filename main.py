from unittest import TestCase
from enum import Enum
import numpy as np
import re


tests = [
    ['L (0, 2,+1) (2, 0, -1) (0, 4, -1) (4, 0, +1) (0, 6, -1) (6, 0, +1)',
     '0.29 0.71 0.14 0.86 0.06 0.94'],
    ['P (0, 2,+1) (2, 0, -1) (0, 4,+1) (4, 0, -1)', '-2.0, 0.0'],
    ['L (0, 2,+1) (2, 0, -1) (0, 4,+1) (4, 0, -1)', '0.98 0.02 1.0 0.0'],
    ['L (0, 2,+1) (2, 0, -1) (0, 4, -1) (4, 0, +1) (0, 6, -1) (6, 0, +1)',
     '0.29 0.71 0.14 0.86 0.06 0.94'],
    ['P (0, 2,+1) (2, 0, -1) (0, 4, -1) (4, 0, +1) (0, 6, -1) (6, 0, +1)', '0.0, -2.0'],
    ['P (0, 2,+1) (2, 0, -1) (0, 4,+1) (4, 0, -1)', '-2.0, 0.0'],
    ['L (8, 12,+1) (2, 4, -1) (0, 5, -1) (2, 9, +1) (4, 7, -1) (5, 3, +1)',
     '0.92 0.64 0.46 0.6 0.76 0.84'],
    ['L (8, 12,+1) (-3, 6, -1) (0, 18, -1) (11, 15, +1) (9, 7, -1) (10, 8, +1)',
     '1.0 0.0 0.01 1.0 1.0 1.0'],
    ['L (8, 2,+1) (-3, 6, -1) (0, 18, -1) (11, 1, +1) (9, 7, -1) (10, 2, +1)',
     '0.99 0.0 0.0 1.0 0.01 1.0'],
    ['L (6, 2,+1) (7, 6, -1) (10, 18, -1) (11, 1, +1) (9, 7, -1) (10, 2, +1)',
     '0.97 0.01 0.0 1.0 0.01 1.0'],
    ['L (16, 2,+1) (7, 6, -1) (8, 5, -1) (11, 1, +1) (9, 7, -1) (10, 2, +1)',
     '1.0 0.0 0.01 1.0 0.0 0.99'],
    ['P (1, -2,+1) (-2, 3, -1) (2, 9,+1) (4, 8, -1)', '-8.0, -6.0'],
    ['P (3, -12,+1) (3, 11, -1) (5, 19,+1) (4, 12, -1) (5, 20, +1) (3, 2, -1)', '6.0, 1.0'],
    ['P (2, 10,+1) (-3, 2, -1) (3, 9,+1) (14, 1, -1) (15, 0, +1) (7, 6, -1) (5, 12, +1) (4, 22, -1)', '-2.0, -10.0']
]


def sig(x):
    return 1/(1 + np.exp(-x))


def dot(arr1, arr2):
    total = 0
    for ind, elem in enumerate(arr1):
        total += (elem * arr2[ind])
    return total


class BinaryClassification(Enum):
    POSITIVE = 1,
    NEGATIVE = -1


class ClassificationType(Enum):
    PERCEPTRON = 0,
    LOG_REG = 1


class BinaryPerceptron:
    def __init__(self, values: str):
        self.weights = [0, 0]
        split_values = values.split('(')[1:]
        for i in range(100):
            for each_value in split_values:
                split_inp = re.sub(r'[+() ]+', '', each_value).split(',')
                classif = int(split_inp[-1])
                x1 = int(split_inp[0])
                x2 = int(split_inp[1])
                computed_y = (self.weights[0] * x1) + (self.weights[1] * x2)
                computed_classif = 1 if computed_y >= 0 else -1
                if computed_classif == classif:
                    continue
                else:
                    self.weights[0] = self.weights[0] + (classif * x1)
                    self.weights[1] = self.weights[1] + (classif * x2)

    def get_weights(self):
        return '{:.1f}, {:.1f}'.format(self.weights[0], self.weights[1])


class LogisticRegression:
    def __init__(self, values: str):
        self.weights: list[float] = [0, 0]  # parameters
        self.learning_rate = 0.1
        split_values = values.split('(')[1:]
        for i in range(100):
            self.weights[0] = self.weights[0] + \
                self.learning_rate * self.compute_log_loss(split_values, 0)
            self.weights[1] = self.weights[1] + self.learning_rate * \
                self.compute_log_loss(split_values, 1)

        print(self.weights)

    def compute_log_loss(self, values: list[str], jth_feature: int):
        the_sum = 0
        for each_value in values:
            split_inp = re.sub(r'[+() ]+', '', each_value).split(',')
            classif = int(split_inp[-1])
            the_sum += (int(classif) - sig(dot(self.weights,
                        [int(split_inp[0]), int(split_inp[1])]))) * int(split_inp[jth_feature])
        return the_sum


def process_inp(inpt: str) -> tuple[ClassificationType, str]:
    spl_inp = inpt.split(' ')
    return (ClassificationType.PERCEPTRON if spl_inp[0] == 'P' else ClassificationType.LOG_REG, inpt[2:])


def main(inp=False):
    if inp:
        processed_inp = process_inp(input())
        if processed_inp[0] == ClassificationType.PERCEPTRON:
            weights = BinaryPerceptron(processed_inp[1]).get_weights()
            print(weights)
    else:
        ut = TestCase()
        for each_test in tests:
            [given, expected] = each_test
            processed_inp = process_inp(given)
            if processed_inp[0] == ClassificationType.PERCEPTRON:
                print(f'----- TESTING {given} -----')
                weights = BinaryPerceptron(processed_inp[1]).get_weights()
                ut.assertEqual(weights, expected)
                print(f'----- PASSED {given} ------')
            else:
                print(f'------ TESTING ------')
                weights = LogisticRegression(processed_inp[1])


if __name__ == '__main__':
    main()
