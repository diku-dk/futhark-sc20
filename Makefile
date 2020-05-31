FUTHARK_VERSION=0.15.8-linux-x86_64
FUTHARK_TARBALL=futhark-$(FUTHARK_VERSION).tar.xz

bin/futhark:
	mkdir -p bin
	tar --strip-components=1 -xf $(FUTHARK_TARBALL) futhark-$(FUTHARK_VERSION)/bin/futhark
