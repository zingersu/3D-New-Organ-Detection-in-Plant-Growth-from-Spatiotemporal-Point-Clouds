"""
Created on Fri Mar  10 12:27:49 2023

@author: YC-W
"""
###########################此程序专门针对PointNet++跑出的结果，使其保存为一个个的txt文件，分别为gt-语义，gt-实例，pred-语义，pred-实例
import numpy as np
import os

def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist

data=np.loadtxt(r"/home/foysal/PhD_Projects/models/dgcnn/dgcnn/part_seg/Epoch472/test_results/out_pred.txt")#获取预测的数据
gtdata=np.loadtxt(r"/home/foysal/PhD_Projects/models/dgcnn/dgcnn/part_seg/Epoch472/test_results/out_gt.txt")#获取gt
pcd=[]
seg=[]
#ins=[]
gt=[]
#gt_ins=[]
gt_seg=[]

Filelist=get_filelist(r"/home/foysal/PhD_Projects/NOS/for_train/test/test")#获取test前的数据的名字，使生成的新文件的名字和训练前的保持一致
##print(Filelist[0][37:])
#print(data)
for i in range(1820):#1820是测试集的大小，根据测试集大小而改变
    pcd=data[i*4096:((i+1)*4096)]
    gt=gtdata[i*4096:((i+1)*4096)]
    data1=pcd[:,:3]
    data2=pcd[:,3]
    seg=np.concatenate((pcd[:,:3],pcd[:,3].reshape(4096,1)),axis=-1)
    gt_seg=np.concatenate((pcd[:,:3],gt[:].reshape(4096,1)),axis=-1)
#    ins=np.concatenate((pcd[:,:3],pcd[:,5].reshape(4096,1)),axis=-1)
#    gt_ins=np.concatenate((pcd[:,:3],gt[:,1].reshape(4096,1)),axis=-1)
 #   np.savetxt(r"F:\plantnet数据server\164_11.7_FPS_10.19\out\ins\\"+Filelist[i][51:],ins,fmt="%f %f %f %d")
    np.savetxt(r"/home/foysal/PhD_Projects/models/dgcnn/dgcnn/results/sem/"+Filelist[i][51:],seg,fmt="%f %f %f %d")
 #   np.savetxt(r"F:\plantnet数据server\164_11.7_FPS_10.19\out\ins_gt\\"+Filelist[i][51:],gt_ins,fmt="%f %f %f %d")
    np.savetxt(r"/home/foysal/PhD_Projects/models/dgcnn/dgcnn/results/gt/"+Filelist[i][51:],gt_seg,fmt="%f %f %f %d")#Filelist[i][37:]根据文件夹名字而改变数字大小，目的是使前后名字一致
    print(i)
