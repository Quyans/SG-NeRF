# 采样脚本，把数据采样出来

from argparse import ArgumentParser
import os
import numpy as np
import shutil


def copyfile(filea,fileb):
    # shutil.copyfile("file.txt","file_copy.txt")
    shutil.copyfile(filea,fileb)

def copydir(dir1,dir2):
    shutil.copytree(dir1,dir2)

def filter_valid_id(posedir,id_list):
        empty_lst=[]
        for id in id_list:
            c2w = np.loadtxt(os.path.join(posedir, "{}.txt".format(id))).astype(np.float32)
            if np.max(np.abs(c2w)) < 30:
                empty_lst.append(id)
        return empty_lst

def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def main():
    parser = ArgumentParser()

    parser.add_argument('--baseSrc', type=str, help='one out of: "train", "test"',
                        default="./data_src/scannet/scans"
    )

    parser.add_argument("--expname", type=str, default="scene0046_00sparse", \
        help='specify the experiment, required for "test" or to resume "train"')

    
    parser.add_argument("--sampleType",type=int,default=0,help="0shows using step,1shows using input xyz")
    parser.add_argument("--step",type=int,default=50,help="0shows using step,1shows using input xyz")

    args = parser.parse_args()
    args.tarname = "{}_sparse".format(args.expname)
    targetDB = os.path.join(args.baseSrc,args.tarname)
    args.targetDB = targetDB
    if os.path.exists(targetDB):
        shutil.rmtree(targetDB)
    makedir(targetDB)


    makedir(os.path.join(targetDB,"exported"))
    
    

    haslabel = False

    colordir = os.path.join(args.baseSrc,args.expname,"exported/color")
    intrinsicdir = os.path.join(args.baseSrc,args.expname,"exported/intrinsic")
    depthdir = os.path.join(args.baseSrc,args.expname,"exported/depth")
    labeldir = os.path.join(args.baseSrc,args.expname,"exported/label")
    posedir = os.path.join(args.baseSrc,args.expname,"exported/pose")


    if  os.path.exists(labeldir):
        haslabel = True

    tarcolordir = os.path.join(args.baseSrc,args.tarname,"exported/color")
    tarintrinsicdir = os.path.join(args.baseSrc,args.tarname,"exported/intrinsic")
    tardepthdir = os.path.join(args.baseSrc,args.tarname,"exported/depth")
    tarlabeldir = os.path.join(args.baseSrc,args.tarname,"exported/label")
    tarposedir = os.path.join(args.baseSrc,args.tarname,"exported/pose")

    tarimgdir = os.path.join(args.baseSrc,args.tarname,"images")

    makedir(tarcolordir)
    # makedir(tarintrinsicdir)
    makedir(tardepthdir)
    makedir(tarlabeldir)
    makedir(tarposedir)

    makedir(tarimgdir)
    

    image_paths = [f for f in os.listdir(colordir) if os.path.isfile(os.path.join(colordir, f))]
    image_paths = [os.path.join(args.baseSrc, args.expname, "exported/color/{}.jpg".format(i)) for i in range(len(image_paths))]
    
    all_id_list = np.array(filter_valid_id(posedir,list(range(len(image_paths)))),dtype="int")
    step=args.step
    train_id_list = all_id_list[::step]
    print("trainid",train_id_list)
    test_id_list = train_id_list[:-1]
    test_id_list = test_id_list + np.ones(len(test_id_list),dtype="int")
    print("testid",test_id_list)
    



    sum_all_id = np.concatenate([train_id_list,test_id_list],dtype="int")
    
    # 替换场景 

    copydir(intrinsicdir,tarintrinsicdir)
    for ind in sum_all_id:
        copyfile(os.path.join(colordir,"{}.jpg".format(ind)),os.path.join(tarcolordir,"{}.jpg".format(ind)))
        copyfile(os.path.join(colordir,"{}.jpg".format(ind)),os.path.join(tarimgdir,"{}.jpg".format(ind)))
        
        if haslabel:
            copyfile(os.path.join(labeldir,"{}.png".format(ind)),os.path.join(tarlabeldir,"{}.png".format(ind)))
        
        copyfile(os.path.join(depthdir,"{}.png".format(ind)),os.path.join(tardepthdir,"{}.png".format(ind)))
        copyfile(os.path.join(posedir,"{}.txt".format(ind)),os.path.join(tarposedir,"{}.txt".format(ind)))

    print("all len",len(sum_all_id))


    # train_dir = os.path.join(targetDB,"train")
    # test_dir = os.path.join(targetDB,"test")
    # for ind in train_id_list:
    #     copyfile(os.path.join(colordir,"{}.jpg".format(ind)),os.path.join(tarcolordir,"{}.jpg".format(ind)))
    #     copyfile(os.path.join(colordir,"{}.jpg".format(ind)),os.path.join(tarimgdir,"{}.jpg".format(ind)))
        
    #     if haslabel:
    #         copyfile(os.path.join(labeldir,"{}.png".format(ind)),os.path.join(tarlabeldir,"{}.png".format(ind)))
        
    #     copyfile(os.path.join(depthdir,"{}.png".format(ind)),os.path.join(tardepthdir,"{}.png".format(ind)))
    #     copyfile(os.path.join(posedir,"{}.txt".format(ind)),os.path.join(tarposedir,"{}.txt".format(ind)))


    with open(os.path.join(targetDB,"imageinfo.txt"),mode="w") as f:

        #写字符串

        # f.write("trainimage len")

        #写字符串或者列表

        f.writelines("trainimage len,{}\n".format(len(train_id_list)))
        f.writelines("testimage len,{}\n".format(len(test_id_list)))
        f.writelines("allLen len,{}\n".format(len(sum_all_id)))


    with open(os.path.join(targetDB,"train_set.csv"),mode="w") as f:

        for ind in train_id_list:
            f.writelines("{}.jpg\n".format(ind))
    
    with open(os.path.join(targetDB,"test_set.csv"),mode="w") as f:

        for ind in test_id_list:
            f.writelines("{}.jpg\n".format(ind))

    # list

    # 

    # self.image_paths = [f for f in os.listdir(colordir) if os.path.isfile(os.path.join(colordir, f))]
    # self.image_paths = [os.path.join(self.data_dir, self.scan, "exported/color/{}.jpg".format(i)) for i in range(len(self.image_paths))]
    # self.all_id_list = self.filter_valid_id(list(range(len(self.image_paths))))
    # if len(self.all_id_list) > 2900: # neural point-based graphics' configuration
    #     # self.test_id_list = self.all_id_list[::100]
    #     # self.train_id_list = [self.all_id_list[i] for i in range(len(self.all_id_list)) if (((i % 100) > 19) and ((i % 100) < 81 or (i//100+1)*100>=len(self.all_id_list)))]
    #     step=self.opt.train_step
    #     self.train_id_list = self.all_id_list[::step]
    #     self.test_id_list = [self.all_id_list[i] for i in range(len(self.all_id_list)) if (i % step) !=0] if self.opt.test_num_step != 1 else self.all_id_list
    # else:  # nsvf configuration
    #     step=self.opt.train_step



    #     self.train_id_list = self.all_id_list[::step]
    #     self.test_id_list = [self.all_id_list[i] for i in range(len(self.all_id_list)) if (i % step) !=0] if self.opt.test_num_step != 1 else self.all_id_list

    # print("all_id_list",len(self.all_id_list))
    # print("test_id_list",len(self.test_id_list), self.test_id_list)
    # print("train_id_list",len(self.train_id_list))
    # self.train_id_list = self.remove_blurry(self.train_id_list)
    # self.id_list = self.train_id_list if self.split=="train" else self.test_id_list
    # self.view_id_list=[]

if __name__ == "__main__":
    main()