BACKEND=cuda
OPTIONS=--pass-option=--nvrtc-option=-arch=compute_35 --pass-option=--default-num-groups=272

all: cudasamples.json nvidiahisto.json

cudasamples.json: cudasamples.fut
	futhark bench cudasamples.fut $(OPTIONS) --backend=$(BACKEND) --json $@

nvidiahisto.json: nvidiahisto.fut
	futhark bench nvidiahisto.fut $(OPTIONS) --backend=$(BACKEND) --json $@
