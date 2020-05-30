# Common Makefile options used by Futhark implementations.

BACKEND=cuda
OPTIONS=--pass-option=--nvrtc-option=-arch=compute_35 --pass-option=--default-num-groups=272 -r 100

%.json: %.fut
	futhark bench $^ $(OPTIONS) --backend=$(BACKEND) --json $@
