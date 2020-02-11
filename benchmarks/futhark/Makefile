BACKEND=cuda
OPTIONS=--pass-option=--nvrtc-option=-arch=compute_35 --pass-option=--default-num-groups=272

prototype.json: prototype.fut
	futhark bench prototype.fut $(OPTIONS) --backend=$(BACKEND) --json $@
