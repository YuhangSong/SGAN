import config
import subprocess
import time
import numpy as np

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