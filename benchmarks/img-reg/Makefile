.NOTPARALLEL:

include ../futhark.mk

export FUTHARK_NUM_GROUPS

all: small-orig.runtime large-orig.runtime small-fut.runtime large-fut.runtime
	@printf "%20s %10s %10s\n" Workload PyTorch Futhark
	@printf "%20s %10sms %10sms\n" "N = 2M, H= 2809" $$(cat small-orig.runtime) $$(cat small-fut.runtime)
	@printf "%20s %10sms %10sms\n" "N = 8M, H=41209" $$(cat large-orig.runtime) $$(cat large-fut.runtime)

%-orig.runtime:
	python3 img-reg-appl-orig.py $*

%-fut.runtime: imgRegHisto.py
	python3 img-reg-appl-wfut.py $*

imgRegHisto.py: imgRegHisto.fut
	$(FUTHARK) pyopencl --library imgRegHisto.fut

.PHONY: clean all

clean:
	rm -f imgRegHisto.py *.pyc *.runtime
