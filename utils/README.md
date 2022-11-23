
<font color='red'>Most steps follows ScanNet\ScanNet in github.</font>
# 1.DOWNLOAD DATASET
## a).Download script from Github 
```
git clone https://github.com/ScanNet/ScanNet.git
```
## b).Download 'download-scannet.py' from BeihangYunPan
## c).Data preparation
The layout should like this,
```
your_dataset_file
├──ScanNet
├──download-scannet.py
```
and change directory to your dataset file:
```
cd your_dataset_file
```
## d).Download the room you want
```
python download-scannet.py -o . --id scene0000_00 
```
Paras explaination: 
* -o: dictory you want to download data
* -id: room id you want to download 

Or just download all the data use the  following command -_-!
```
python download-scannet.py -o scans
```
**Attention: Several sequences are often included in one room. So id scene0101_04 means room0101 with sequence04** 
# 2.TRANSFORM .sens TO .PNG\ .JPG\ .TXT 
Native datasetes are organized by .sens, use the following command to transform it to png/jpg/txt.
```
python ScanNet/SensReader/python/reader.py --filename scans/scene0000_00/scene0000_00.sens --output_path /home/slam/devdata/pointnerf/data_src/scannet/scans/scene0000_00/exported/ --export_depth_images --export_color_images --export_poses --export_intrinsics
```
Paras explanation: 
* --filename: path of .sens file
* --output_path: path you want to export(You should create by yourself.)
* --export_depth_images: opentional to export depth image
* --export_color_images: opentional to export color image
* --export_poses : opentional to export camera pose
* --export_intrinsics: opentional to export camera intrinsics

# 3.Data Organization
*Coming soon...*
# 4.Just Enjoy ScanNet!(\~_\~)
