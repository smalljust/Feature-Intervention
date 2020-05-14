# https://zh.d2l.ai/chapter_optimization/momentum.html
import numpy as np
from utils.timer import Timer
def ff(x,args):
    return np.array([x[0][0] ** 2 + x[0][1] ** 2+args[0],
                     0.1 * x[0][0] ** 2 + 2 * x[0][1] ** 2+args[0]])

class myOptimizer:
    def __init__(self, f, x,*args, type="RMSProp", cfg=None):
        self.f = f
        self.x = x
        self.args=args
        self.s = np.zeros(shape=x.shape)
        self.opt_func = self.__getattribute__(type)
        self.cfg = cfg if cfg else {}
        self.best = f(x,self.args)
        self.best_x = x.copy()

    def cal_gradient(self):
        delta = self.cfg.get('delta', 1e-6)
        gradients = np.zeros(shape=self.x.shape)
        for idx in range(self.x.shape[-1]):
            x1 = self.x.copy()
            x1[:, idx] = self.x[:, idx] + delta
            x2 = self.x.copy()
            x2[:, idx] = self.x[:, idx] - delta
            gradients[:, idx] = (self.f(x1,self.args) - self.f(x2,self.args)) / delta / 2
        return gradients

    def arg_min(self):
        epoch = self.cfg.get('epoch', 100)
        #timer=Timer()
        for i in range(epoch):
            #timer.start()
            self.opt_func()
            now = self.f(self.x,self.args)
            mask = now < self.best
            self.best[mask] = now[mask].copy()
            self.best_x[mask] = self.x[mask].copy()
            #timer.stop()
        #print(self.x)

    def GD(self):
        lr = self.cfg.get('lr', 0.1)
        gradients = self.cal_gradient()
        self.x -= lr * gradients

    def momentum(self):
        lr = self.cfg.get('lr', 0.1)
        gamma = self.cfg.get('gamma', 0.9)
        gradients = self.cal_gradient()
        self.s = gamma * self.s + lr * gradients
        self.x -= self.s

    def RMSProp(self):
        lr = self.cfg.get('lr', 0.1)
        gamma = self.cfg.get('gamma', 0.9)
        eps = self.cfg.get('eps', 1e-6)
        g = self.cal_gradient()
        self.s = gamma * self.s + (1 - gamma) * g * g
        self.x -= lr / np.sqrt(self.s + eps) * g


if __name__ == '__main__':
    opt = myOptimizer(ff, np.array([[1., 1.], [1., 1.]]),1, cfg={'epoch': 100, 'lr': 0.1})
    opt.arg_min()
    print(opt.best_x)
