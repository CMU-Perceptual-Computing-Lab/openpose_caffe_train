# Official Caffe to OpenPose Custom Caffe

Current Caffe version: Last commit of Apr 18th, 2018
https://github.com/BVLC/caffe



### Caffe Modification
In order to change the official Caffe to the OpenPose version:

1. Modified file(s) (search for `OpenPose` to find the editions inside each file):
    - Makefile
    - src/caffe/proto/caffe.proto
    - include/caffe/util/cudnn.hpp (fixed a warning for cuDNN 7 that current Caffe has not solved yet)
2. New folder(s):
    - autocompile/
    - include/caffe/openpose/
    - src/caffe/openpose/
3. Deleted:
	- Makefile.config.example



### Compilation
Assuming you have all the Caffe prerequisites installed, compile this custom Caffe:

```
cd autocompile/
bash compile_caffe.sh
```
