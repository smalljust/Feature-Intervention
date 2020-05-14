import multiprocessing as mp
import platform
if platform.system() != "Linux":
    print("Windows")
    mp.freeze_support()

class MultiProcess:
    MANAGER = mp.Manager()
    P_Data_List = MANAGER.list()

    class MultiProducer(mp.Process):
        def __init__(self, idx, func, *args):
            super().__init__()
            self.func = func
            self.idx = idx
            self.args = args

        def run(self):
            res = self.func(*self.args)
            MultiProcess.P_Data_List.append((self.idx, res))

    def __init__(self):
        if self.P_Data_List:
            self.P_Data_List.clear()
        self.subs = []

    def add(self, idx, func, *args):
        self.subs.append(self.MultiProducer(idx, func, *args))

    def get_res(self):
        for sub in self.subs:
            sub.run()
        for sub in self.subs:
            sub.join()
        res=[]
        for _,x in sorted(self.P_Data_List,key=lambda x:x[0]):
            res.append(x)
        return res
