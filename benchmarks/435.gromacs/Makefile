run:
	futhark bench --backend=cuda --pass-option=--nvrtc-option=-arch=compute_30 --pass-option=--default-num-groups=272 --pass-option=--default-group-size=256 -r 5000 inl1100-6perit.fut
	futhark bench --backend=opencl --pass-option=--default-num-groups=272 --pass-option=--default-group-size=256 -r 5000 inl1100-withscan.fut
