BACKEND=cuda
OPTIONS=--pass-option=--nvrtc-option=-arch=compute_35 --pass-option=--default-num-groups=272 -r 100

all: cudasamples.json prototype.json

prototype.json: prototype.fut
	futhark bench prototype.fut $(OPTIONS) --backend=$(BACKEND) --json $@

cudasamples.json: cudasamples.fut
	futhark bench cudasamples.fut $(OPTIONS) --backend=$(BACKEND) --json $@
