all:
	$(MAKE) -C src/
	mkdir -p bin
	mv src/correlation bin/

clean:
	rm -rf bin
