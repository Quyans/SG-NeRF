This is the officially implemented of paper "SG-NeRF: Semantic-guided Point-based Neural Radiance Fields"




### Light field browser GUI

Attention！！！
相机插值功能基于 mitsuba2 开发，只能是mitsuba2，不能是mitsuba3

要求 gcc>=8.4
步骤：

https://mitsuba2.readthedocs.io/en/latest/src/getting_started/compiling.html

`bash
export CC=clang-9
export CXX=clang++-9

mkdir build
cd build
cmake -GNinja ..
ninja

#然后将接下来的代码加入~/.bashrc
MITSUBA_DIR=/home/vr717/Documents/qys/code/mitsuba2
MITSUBA_BUILD_DIR="build"
export PYTHONPATH="$MITSUBA_DIR/dist/python:$MITSUBA_DIR/$MITSUBA_BUILD_DIR/dist/python:$PYTHONPATH"
export PATH="$MITSUBA_DIR/dist:$MITSUBA_DIR/$MITSUBA_BUILD_DIR/dist:$PATH"
`

如果不需要相机插值，只需要注释掉前面的,并注释掉报错的方法
`python
import mitsuba
mitsuba.set_variant('scalar_rgb')
from mitsuba.core import ScalarTransform4f, AnimatedTransform
`


Code is based on awesome Point-NeRF.
```
@article{xu2022point,
  title={Point-NeRF: Point-based Neural Radiance Fields},
  author={Xu, Qiangeng and Xu, Zexiang and Philip, Julien and Bi, Sai and Shu, Zhixin and Sunkavalli, Kalyan and Neumann, Ulrich},
  journal={arXiv preprint arXiv:2201.08845},
  year={2022}
}
```
