# standard optimisation
CFLAGS=-O -D_FILE_OFFSET_BITS=64

#GPUFLAGS=-O2 --ptxas-options=-v -arch=sm_50

# set default install directory
prefix = $(HOME)
INSTALL=install

TARGETS=corr_gpu_complex

all: $(TARGETS)

install: $(TARGETS)
	mkdir -p $(prefix)/bin
	$(INSTALL) $(TARGETS) $(prefix)/bin

uninstall:
	$(foreach target,$(TARGETS),rm -f $(prefix)/bin/$(target);)

corr_gpu_complex: corr_gpu_complex.cu
	nvcc $(GPUFLAGS) -o corr_gpu_complex corr_gpu_complex.cu -lcufft
	chmod go+rx corr_gpu_complex

clean:
	rm -f $(TARGETS)
