import config
import subprocess
import time
import numpy as np

def fake_gsa():

    try:
        path_dic_of_requiring_file=GetFileList(FindPath=config.real_state_dir,
                                           FlagStr=['__requiring.npz'])
    except Exception, e:
        print(str(Exception)+": "+str(e))
        return

    for i in range(len(path_dic_of_requiring_file)):

        path_of_requiring_file = path_dic_of_requiring_file[i]

        if path_of_requiring_file.split('.np')[1] is not 'z':
            print(path_of_requiring_file)
            print('pass')
            continue

        try:
            requiring_state = np.load(path_of_requiring_file)['state']
        except Exception, e:
            print(str(Exception)+": "+str(e))
            continue

        subprocess.call(["mv", path_of_requiring_file, path_of_requiring_file.split('__')[0]+'__done.npz'])

        file = config.waiting_reward_dir+path_of_requiring_file.split('/')[-1].split('__')[0]+'__waiting.npz'
        np.savez(file,
                 gsa_reward=[0.32])

def IsSubString(SubStrList,Str):

    flag=True
    for substr in SubStrList:
        if not(substr in Str):
            flag=False

    return flag

def GetFileList(FindPath,FlagStr=[]):

    import os
    FileList=[]
    FileNames=os.listdir(FindPath)
    if (len(FileNames)>0):
       for fn in FileNames:
           if (len(FlagStr)>0):
               if (IsSubString(FlagStr,fn)):
                   fullfilename=os.path.join(FindPath,fn)
                   FileList.append(fullfilename)
           else:
               fullfilename=os.path.join(FindPath,fn)
               FileList.append(fullfilename)

    if (len(FileList)>0):
        FileList.sort()

    return FileList
