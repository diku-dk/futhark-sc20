include ../futhark.mk

run:
	futhark bench --backend=$(BACKEND) $(OPTIONS) -r 5000 inl1100-6perit.fut --json inl1100-6perit.json
	futhark bench --backend=$(BACKEND) $(OPTIONS) -r 5000 inl1100-withscan.fut --json inl1100-withscan.json
