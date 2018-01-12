all: lib-ugly live upload
lazy: live upload

lib:
	arm-linux-gnueabihf-g++ -o build/examples/squ.o -c -Wno-deprecated-declarations -Wall -DARCH_ARM -Wextra -Wno-unused-parameter -pedantic -Wdisabled-optimization -Wformat=2 -Winit-self -Wstrict-overflow=2 -Wswitch-default -fpermissive -std=gnu++11 -Wno-vla -Woverloaded-virtual -Wctor-dtor-privacy -Wsign-promo -Weffc++ -Wno-format-nonliteral -Wno-overlength-strings -Wno-strict-overflow -Wlogical-op -Wnoexcept -Wstrict-null-sentinel -march=armv7-a -mthumb -mfpu=neon -mfloat-abi=hard -fPIC -Werror -O3 -ftree-vectorize -D_GLIBCXX_USE_NANOSLEEP -DARM_COMPUTE_CPP_SCHEDULER=1 -Iinclude -I. -I. examples/squeezenet_lib.cpp

lib-ugly:
	arm-linux-gnueabihf-g++ -o build/examples/squ.o -c -Wall -DARCH_ARM -fpermissive -std=gnu++11 -march=armv7-a -mthumb -mfpu=neon -mfloat-abi=hard -fPIC -O3 -ftree-vectorize -D_GLIBCXX_USE_NANOSLEEP -DARM_COMPUTE_CPP_SCHEDULER=1 -Iinclude -I. -I. examples/squeezenet_lib.cpp

live:
	arm-linux-gnueabihf-g++ -o build/examples/squ_live -static-libgcc -static-libstdc++ -lpthread examples/squeezenet_live.cpp build/examples/squ.o build/utils/Utils.o -Lbuild -L. -Lexamples/ -Lbuild/opencl-1.2-stubs -Lopencl-1.2-stubs build/libarm_compute-static.a build/libarm_compute_core-static.a examples/libvisionloader.so.0 -Wl,--allow-shlib-undefined

upload:
	scp build/examples/squ_live root@192.168.0.90:/tmp
