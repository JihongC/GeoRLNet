import random
import sys
from tqdm import tqdm

from netEnv import NetEnv


class Test():
    def __init__(self):
        self.env = NetEnv()
        self.env.reset()

    def random(self):
        s = 0
        rew = []
        for i in range(10000):
            obs, r, done, _ = self.env.step(random.choice([0, 1, 2]))
            if done:
                break
            s += r
            rew.append(r)
        return r, s, s / len(rew)

    def pooling(self):
        s = 0
        rew = []
        for i in range(10000):
            obs, r, done, _ = self.env.step(i % 3)
            if done:
                break
            s += r
            rew.append(r)
        return r, s, s / len(rew)

    def choose_one(self):
        s = 0
        rew = []
        for i in range(10000):
            obs, r, done, _ = self.env.step(0)
            if done:
                break
            s += r
            rew.append(r)
        return r, s, s / len(rew)

    def test(self, test_way='random'):
        sum = 0
        max = -sys.maxsize
        if test_way == 'random':
            for i in tqdm(range(200)):
                self.env.reset()
                _, s, _ = self.random()
                sum += s
                if s > max:
                    max = s
            return sum / 200, max
        if test_way == 'pooling':
            for i in tqdm(range(200)):
                self.env.reset()
                _, s, _ = self.pooling()
                sum += s
                if s > max:
                    max = s
            return sum / 200, max
        if test_way == 'choose_one':
            for i in tqdm(range(200)):
                self.env.reset()
                _, s, _ = self.choose_one()
                sum += s
                if s > max:
                    max = s
            return sum / 200, max
