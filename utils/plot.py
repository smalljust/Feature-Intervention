import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

def plot(name):
    f, ax1 = plt.subplots(figsize=(6, 6), nrows=1)
    #plt.title('Pearson Correlation of Features', y=1.05, size=15)
    with open(name,'r') as f:
        data_js=json.load(f)
    origin = 0
    feature_set=[]
    data=np.zeros((6,6))
    for k,v in data_js.items():
        if k.find("nomal")>=0:
            origin=v
            continue
        features=k.strip().split(' ')
        idxs=[]
        for feature in features:
            if feature not in feature_set:
                feature_set.append(feature)
            idxs.append(feature_set.index(feature))
        if len(idxs)==1:
            idxs=idxs+idxs
        if len(idxs)==2:
            data[idxs[0]][idxs[1]]=(origin-v)/origin
            data[idxs[1]][idxs[0]] =(origin - v)/origin
    #print(data)
    sns.heatmap(data, linewidths=0.1, vmax=1.0,square=True, linecolor='white', annot=True)
    ax1.set_title('the decline proportion of acc after shuffle feature')
    ax1.set_xlabel('Features')
    ax1.set_xticklabels(feature_set)
    ax1.set_ylabel('Features')
    ax1.set_yticklabels(feature_set)
    plt.show()


def plot_dis(source_dis:dict,target_dis:dict,name):
    x = list(source_dis.keys())
    y1=[]
    y2=[]
    for feature in x:
        y1.append(100)
        y2.append(source_dis[feature]/(source_dis[feature]+target_dis[feature])*100)
    plt.figure(figsize=(15, 8))
    plt.bar(x, y1, label="label1", color='red')
    plt.bar(x, y2, label="label2", color='orange')

    plt.xticks(np.arange(len(x)), x, rotation=90, fontsize=10)  # 数量多可以采用270度，数量少可以采用340度，得到更好的视图
    plt.yticks(range(0, 110, 10), range(0, 110, 10), rotation=90, fontsize=10)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.savefig(name+".png")


def plot_feature(acc,name):
    y1=acc['tot']
    plt.figure(figsize=(8,3))
    plt.plot([0,7],[y1,y1],color='red')
    x=[]
    y=[]
    for k,v in acc.items():
        try:
            x.append(int(k)/48)
            y.append(v)
        except:
            pass
    plt.scatter(x,y)
    #plt.show()
    plt.savefig(name+".png")

def plot_normal(avg,std,minn,maxn):
    x = np.arange(minn,maxn, 0.1)
    y = np.exp(-(x-avg)**2/2/std**2)/np.sqrt(2*np.pi)/std
    plt.plot(x,y)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_2d_normal(avg,std,minn,maxn,num=200):
    def cal_Z():
        pos = np.concatenate((np.expand_dims(X, axis=2), np.expand_dims(Y, axis=2)), axis=2)  # 定义坐标点
        print(pos.shape)
        pos=pos.reshape(num*num,2)
        a = np.dot((pos - u), np.linalg.inv(o))  # o的逆矩阵
        b = np.expand_dims(pos - u, axis=3)
        Z = np.zeros((num, num), dtype=np.float32)
        #for i in range(num):
        #    Z[i] = [np.dot(a[i, j], b[i, j]) for j in range(num)]  # 计算指数部分

        Z = np.exp(Z * (-0.5)) / (2 * np.pi * math.sqrt(np.linalg.det(o)))
        return Z
    import math
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import cm
    import matplotlib as mpl
    l = np.linspace(minn, maxn, num)
    X, Y = np.meshgrid(l, l)

    u = np.array(avg)  # 均值
    o= np.diag(std)**2
    #o = np.array([[1, 0.5], [0.5, 1]])  # 协方差矩阵

    Z = np.exp(-0.5*((X-avg[0])**2/std[0]/std[0]+(Y-avg[1])**2/std[1]/std[1]))/2/np.pi/std[0]/std[1]


    fig = plt.figure()  # 绘制图像
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.6, cmap=cm.rainbow)


    #cset = ax.contour(X, Y, Z, 10, zdir='z', offset=0, cmap=cm.coolwarm)  # 绘制xy面投影
    Z1= np.exp(-0.5 * ((Y - avg[1]) ** 2 / std[1] / std[1])) / np.sqrt(2 * np.pi) / std[1]
    cset = ax.contour(X, Y, Z1, zdir='x', offset=-4, cmap=mpl.cm.winter)  # 绘制zy面投影
    #ax.plot(X[:,0],Y[:,int(num/2)],Z[:,int(num/2)])
    Z2 = np.exp(-0.5 * ((X - avg[0]) ** 2 / std[0] / std[0])) / np.sqrt(2 * np.pi) / std[0]
    cset = ax.contour(X, Y, Z2, zdir='y', offset=4, cmap=mpl.cm.winter)  # 绘制zx面投影
    #ax.plot(X[int(num/2),:], Y[num-1,:], Z[int(num / 2)])
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')

    plt.show()

if __name__ == '__main__':
    avg=[0,0.5]
    std=[1,0.5]
    minn=min(avg[0]-3*std[0],avg[1]-3*std[1])
    maxn=max(avg[0]+3*std[0],avg[1]+3*std[1])
    plot_2d_normal(avg,std,minn,maxn)
    #plot_normal(avg[1],std[1],minn,maxn)

