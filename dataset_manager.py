import os 
import xml.etree.ElementTree as ET
import random
import shutil
import cv2 as cv
import numpy as np
from PIL import Image, ImageStat
IMG_TYPE = ['png','jpg','bmp']
NORMAL_TYPE = ['png','jpg','bmp','txt','xml']
dic = {'armor_sentry_blue':0,   #BG
       'armor_sentry_red':1,     #B1
       'armor_sentry_none':2,    #B2
       'armor_hero_blue':3,      #B3
       'armor_hero_red':4,       #B4
       'armor_hero_none':5,      #B5
       'armor_engine_blue':6,    #BO
       'armor_engine_red':7,     #BBs
       'armor_engine_none':8,     #BBb
       'armor_infantry_3_blue':9,  #RG
       'armor_infantry_3_red':10,  #R1
       'armor_infantry_3_none':11,  #R2
       'armor_infantry_4_blue':12,  #R3
       'armor_infantry_4_red':13,    #R4
       'armor_infantry_4_none':14,    #R5
       'armor_infantry_5_blue':15,  #RO
       'armor_infantry_5_red':16,   #RBs
       'armor_infantry_5_none':17,  #RBb
       'armor_outpost_blue':18,      #NG
       'armor_outpost_red':19,  #N1
       'armor_outpost_none':20,  #N2
       'armor_base_blue':21,   #N3
       'armor_base_red':22,    #n4
       'armor_base_purple':23,
       'armor_base_none':24,
       }
r_dic = {v:k for k,v in dic.items()}
big_dic = {
    "25":"23",
    "26":"24",
    "27":"25",
    "28":"26",
    "29":"27",
    "30":"28",
}
def loadfolder(path,target_suffix):
    target = []
    for dirpath, dirnames,filenames in os.walk(path):
        for fn in filenames:
            suf =fn.split('.')[-1]
            if suf in target_suffix:
                target.append((dirpath,fn))
            elif suf not in NORMAL_TYPE:
                print(dirpath+'/'+fn)
    return target

def remove012(files):
    for (p,n) in files:
        d = []
        file = p+'/'+n
        with open(file, 'r', encoding='utf-8') as f:
            datas = f.readlines()
            for data in datas:
                if data.split(' ')[0] not in ['0','1','2']:
                    d.append(data)
        f.close()
        with open(file, 'w') as writers: # 打开文件
            for i in d:
                writers.write(i)
        writers.close()

def convertBigNum(files): 
    for (p,n) in files:
        d = []
        file = p+'/'+n
        with open(file, 'r', encoding='utf-8') as f:
            datas = f.readlines()
            for data in datas:
                line = data.split(' ')
                try:
                    line[0] =big_dic[line[0]]
                except  Exception:
                    d.append(data)
                    continue
                data = " ".join(line)
                d.append(data)
        f.close()
        with open(file, 'w') as writers: # 打开文件
            for i in d:
                writers.write(i)
        writers.close()

def clamp(str_):
    if float(str_)<0:
        return str(0)
    elif float(str_)>1:
        return str(1)
    else:
        return str_
    
def copyPaste(src,src_label,dst,dst_labels):
   src = cv.imread(src)
   dst = cv.imread(dst)
   msk = np.zeros(dst.shape[:2],np.uint8)
   src_shape = np.asarray(src.shape[:2] * 4)[[1,0,3,2,5,4,7,6]]
   dst_shape = np.asarray(dst.shape[:2] * 4)[[1,0,3,2,5,4,7,6]]
   src_label = [float(x) for x in src_label.split(' ')[1:]]* src_shape
  
   max_x = int(np.max(src_label[[0,2,4,6]]))
   min_x = int(np.min(src_label[[0,2,4,6]]))
   max_y = int(np.max(src_label[[1,3,5,7]]))
   min_y = int(np.min(src_label[[1,3,5,7]]))
   src = src[min_y:max_y,min_x:max_x]
   src_label = src_label - [min_x,min_y]*4

   pst1=np.float32([[src_label[0],src_label[1]],[src_label[2],src_label[3]],[src_label[4],src_label[5]],[src_label[6],src_label[7]]])
   rmsks = []
   tmsks = []
   for dst_label in dst_labels:
        dst_label = [float(x) for x in dst_label.split(' ')[1:]]* dst_shape

        pst2=np.float32([[dst_label[0],dst_label[1]],[dst_label[2],dst_label[3]],[dst_label[4],dst_label[5]],[dst_label[6],dst_label[7]]])
        matrix=cv.getPerspectiveTransform(pst1,pst2)
        msk=cv.warpPerspective(src,matrix,(dst.shape[1],dst.shape[0]))
        gmsk = cv.cvtColor(msk, cv.COLOR_BGR2GRAY)
        _,temp_msk= cv.threshold(gmsk, 0, 255,cv.THRESH_BINARY)
        rmsks.append(cv.bitwise_not(temp_msk))
        tmsks.append(msk)
   for i,tmsk in enumerate(tmsks):
        dst = cv.merge([cv.bitwise_and(s,rmsks[i]) for s in cv.split(dst)])
        dst = cv.add(dst,tmsk)
   return dst
    
    

  
   
def xml2txt(files):
    for (p,n) in files:
        xmlfilename = p+'/'+n
        tree = ET.ElementTree(file=xmlfilename)
        node = tree.getroot()
        objects =node.findall("object")
        txt_contents =[]

        for obj in objects:
            name = str(dic[obj.find("name").text])

            if name in ["0","1","2","23","24"]:
                continue
            node = obj.find("bndbox")
            top_left_x = clamp(node.find("top_left_x").text)
            top_left_y = clamp(node.find("top_left_y").text)
            bottom_left_x =  clamp(node.find("bottom_left_x").text)
            bottom_left_y =  clamp(node.find("bottom_left_y").text)
            bottom_right_x =  clamp(node.find("bottom_right_x").text)
            bottom_right_y =  clamp(node.find("bottom_right_y").text)
            top_right_x =  clamp(node.find("top_right_x").text)
            top_right_y =  clamp(node.find("top_right_y").text)
            txt_contents.append(" ".join([name,top_left_x,top_left_y,bottom_left_x,bottom_left_y,bottom_right_x,bottom_right_y,top_right_x,top_right_y])+'\n')
        file = open(p+'/'+n.split('.')[-2]+'.txt','w')
        file.writelines(txt_contents)  
        file.close()

def removexml(files):
    for (p,n) in files:
        os.remove(p+'/'+n)   


def removenone(files):
    for (p,n) in files:
        file = p+'/'+n
        with open(file, 'r', encoding='utf-8') as f:
            data = f.readlines()
            if len(data) == 0:
                f.close()
                os.remove(p+'/'+n)
                try:
                    os.remove(p+'/'+txt2png(n)) 
                except  Exception:
                    os.remove(p+'/'+txt2jpg(n)) 
                continue  
        f.close()    

def saveLabel(path,name,img,labels):
     with open(path+'/'+name+'.txt', 'w') as writers: 
            for l in labels:
                writers.write(l)
     writers.close()
     cv.imwrite(path+'/'+name+'.png',img)


def makeSentry(sentry_path,oths_path,num,dst):
    sts = loadfolder(sentry_path,['txt'])
    oths = loadfolder(oths_path,['txt'])
    if  num>len(oths):
        print("太多了"+str(len(oths)))
        return 
    
    random.shuffle(sts)
    random.shuffle(oths)
    for x in range(num):
        st,stn = sts[x% len(sts)]
        oth,othn = oths[x]
        with open(st+'/'+stn, 'r', encoding='utf-8') as f:
            stl = f.readlines()[0]
        f.close()
        datas =[ ]
        ori_dta = []
        with open(oth+'/'+othn, 'r', encoding='utf-8') as f:
            temp_datas = f.readlines()
            for l in temp_datas:
                if l.split(' ')[0] in ['23','24','25','26','27','28','21','22','3','4','5']:
                    ori_dta.append(l)
                else:
                    datas.append(l)
        f.close()
        try:
            res =  copyPaste(st+'/'+txt2jpg(stn), stl ,oth+'/'+txt2png(othn),datas)
        except Exception:
            res =  copyPaste(st+'/'+txt2jpg(stn), stl ,oth+'/'+txt2jpg(othn),datas)
        cv.imwrite(dst+'/'+str(x)+".png",res)
        for d in datas:
            ld = d.split(' ')
            ld[0] = stl[0]
            ori_dta.append(' '.join(ld))
        
        file = open(dst+'/'+str(x)+".txt",'w')
        file.writelines(ori_dta)  
        file.close()
        


def res_viewer(path,gpath,bpath):
    txts = loadfolder(path,['txt'])
    gcnt =0
    bcnt =0
    for tp,tn in txts:
        imgp = tp+'/'+tn.split('.')[-2]+'.png' 
        img0 = cv.imread(imgp)
        img = cv.resize(img0,(img0.shape[1]*2,img0.shape[0]*2))

        with open(tp+'/'+tn, 'r', encoding='utf-8') as f:
            datas = f.readlines()
            for data in datas:
                line = data.split(' ')
                name = r_dic[int(line[0])]
                loc = [float(x) for x in line[1:]]* np.asarray(img.shape[:2] * 4)[[1,0,3,2,5,4,7,6]] 
                loc = [int(x) for x in loc]
                img = cv.circle(img, (loc[0],loc[1]), 2, (0,255,255), 2) 
                img = cv.circle(img, (loc[2],loc[3]), 2, (255,255,0), 2) 
                img = cv.circle(img, (loc[4],loc[5]), 2, (0,255,), 2) 
                img = cv.circle(img, (loc[6],loc[7] ), 2, (0,0,255), 2) 
                cv.putText(img,name,(loc[2],loc[3]+5),cv.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),1)

            cv.imshow("show", img)
            key = cv.waitKey(0) 
            print(key)
            if key == 108:
            #    saveLabel(bpath,str(bcnt),img0,datas)
                bcnt += 1
            elif key ==120:
                os.remove(tp+'/'+tn)   
                os.remove(imgp)   

            else:  
            #    saveLabel(gpath,str(gcnt),img0,datas)
                gcnt += 1
                
        f.close()





def png2txt(s):
    return s.split('.')[-2]+'.txt' 
def txt2png(s):
    return s.split('.')[-2]+'.png' 
def txt2jpg(s):
    return s.split('.')[-2]+'.jpg' 

if __name__ == '__main__':
    res_viewer("./res","./good","./bad")
    # sourcepath =  "./xmlDataset"
    # datasetpath = "./detaset"
#     src = "./xmlDataset/23sentry/HERO-23-OTH-0.jpg" 
#     src_label="1 0.423553 0.526173 0.420942 0.551289 0.46648 0.555357 0.468751 0.530057"
#     dst = "./xmlDataset/SJTU/SJTU-21/UC_N/Armor/SJTU-21-UC_N-3213.png"
#     dst_label = ["14 0.664582 0.517403 0.665947 0.562215 0.73865 0.560478 0.736923 0.513947",
# "7 0.593877 0.465479 0.593162 0.491937 0.63035 0.492271 0.630407 0.465778",
# "22 0.0342471 0.346396 0.0330575 0.365014 0.0753888 0.369237 0.0755143 0.349704",
# "5 0.760138 0.488119 0.759425 0.513455 0.834388 0.50855 0.835429 0.482262"
# ]
   # makeSentry("xmlDataset/23sentry","./detaset/train",5000,"./temp")
    
    #bigpath = "./balanced_infantry/ann"
   # png = loadfolder(sourcepath,['png','jpg'])
    #xml = loadfolder(sourcepath,['xml'])
   # xml2txt(xml)
    # txt = loadfolder(datasetpath,['txt'])
    #addisbig(txt)
    # convertBigNum(txt)
    #removenone(txt)
    # bigtxt = loadfolder(bigpath,['txt'])
    #  convertBigNum(bigtxt)


    # png = loadfolder(sourcepath,['png','jpg'])
    # random.seed(2024116)
    # random.shuffle(png)
    # thre = len(png)*0.8
    # for i,(p,n) in enumerate(png):
    #     if i<thre:
    #         try:
    #             shutil.copy(p+'/'+n,datasetpath+"/train/"+n)
    #             shutil.copy(p+'/'+png2txt(n),datasetpath+"/train/"+png2txt(n))
    #         except Exception:
    #             print("未找到"+p+'/'+n)
    #             continue
    #     else:
    #         try:
    #             shutil.copy(p+'/'+n,datasetpath+"/val/"+n)
    #             shutil.copy(p+'/'+png2txt(n),datasetpath+"/val/"+png2txt(n))
    #         except Exception:
    #             print("未找到"+p+'/'+n)
    #             continue
    
