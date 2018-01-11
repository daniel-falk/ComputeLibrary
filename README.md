# SqueezeNet for NEON

Fork of ARM Compute Library with a SqueezeNet implementation that is not using OpenCL! Works perfect on Raspberry PI.

# Instructions:

If cross-compiling, follow steps 1 to 5, if on a neon chip just use native compiler to build, copy images/labels/weights and run.

1. Add "deb http://emdebian.org/tools/debian/ jessie main" to apt sources

2. Add the archive-key
```bash
curl http://emdebian.org/tools/debian/emdebian-toolchain-archive.key | sudo apt-key add -
```

3. Install cross-compilers
```bash
sudo dpkg --add-architecture armhf
sudo apt-get update
sudo apt-get install crossbuild-essential-armhf
```

4. Build
```bash
cd ComputeLibrary
export CXX=arm-linux-gnueabihf-g++
scons Werror=1 debug=0 asserts=0 neon=1 opencl=0 os=linux arch=armv7a examples=1 standalone=1
```
the standalone flag will statically link the whole ACL into the example binaries to simplify move of files. 

5. Run SqueezeNet
```bash
cp -r examples/{images,labels,weights} build/examples
scp -r build/examples/ <arm-machine-ip>:/tmp
ssh <arm-machine-ip>
/tmp/example/neon_squeezenet
```

# Arm Compute Library
Please report issues here: https://github.com/ARM-software/ComputeLibrary/issues
Make sure you are using the latest version of the library before opening an issue. Thanks

Related projects:

- [Caffe on Compute Library](https://github.com/OAID/caffeOnACL)
- [Tutorial: Cartoonifying Images on Raspberry Pi with the Compute Library](https://community.arm.com/graphics/b/blog/posts/cartoonifying-images-on-raspberry-pi-with-the-compute-library)

Documentation available here:

- [v17.12](https://arm-software.github.io/ComputeLibrary/v17.12/)
- [v17.10](https://arm-software.github.io/ComputeLibrary/v17.10/)
- [v17.09](https://arm-software.github.io/ComputeLibrary/v17.09/)
- [v17.06](https://arm-software.github.io/ComputeLibrary/v17.06/)
- [v17.05](https://arm-software.github.io/ComputeLibrary/v17.05/)
- [v17.04](https://arm-software.github.io/ComputeLibrary/v17.04/)
- [v17.03.1](https://arm-software.github.io/ComputeLibrary/v17.03.1/)

Binaries available here:

- [v17.12](https://github.com/ARM-software/ComputeLibrary/releases/download/v17.12/arm_compute-v17.12-bin.tar.gz)
- [v17.10](https://github.com/ARM-software/ComputeLibrary/releases/download/v17.10/arm_compute-v17.10-bin.tar.gz)
- [v17.09](https://github.com/ARM-software/ComputeLibrary/releases/download/v17.09/arm_compute-v17.09-bin.tar.gz)
- [v17.06](https://github.com/ARM-software/ComputeLibrary/releases/download/v17.06/arm_compute-v17.06-bin.tar.gz)
- [v17.05](https://github.com/ARM-software/ComputeLibrary/releases/download/v17.05/arm_compute-v17.05-bin.tar.gz)
- [v17.04](https://github.com/ARM-software/ComputeLibrary/releases/download/v17.04/arm_compute-v17.04-bin.tar.gz)
- [v17.03.1](https://github.com/ARM-software/ComputeLibrary/releases/download/v17.03.1/arm_compute-v17.03.1-bin.tar.gz)

Support: developer@arm.com

License & Contributions: The software is provided under MIT license. Contributions to this project are accepted under the same license.
