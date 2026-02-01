# Makefile for ImageStacker Linux Package
# Usage: make rpm

.PHONY: help rpm install clean

help:
	@echo "ImageStacker Linux Packaging"
	@echo "============================"
	@echo ""
	@echo "Available targets:"
	@echo "  make rpm       - Build RPM package for SUSE"
	@echo "  make install   - Build and install RPM package"
	@echo "  make clean     - Clean build artifacts"
	@echo "  make help      - Show this help"
	@echo ""

rpm:
	@echo "Building RPM package..."
	./build-rpm.sh

install: rpm
	@echo "Installing RPM package..."
	@RPM_FILE=$$(ls -t ~/rpmbuild/RPMS/x86_64/imagestacker-*.rpm 2>/dev/null | head -1); \
	if [ -z "$$RPM_FILE" ]; then \
		echo "Error: RPM package not found!"; \
		exit 1; \
	fi; \
	sudo zypper install -y "$$RPM_FILE"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf ~/rpmbuild/BUILD/imagestacker-*
	rm -f ~/rpmbuild/SOURCES/imagestacker-*.tar.gz
	rm -f ~/rpmbuild/SPECS/imagestacker.spec
	@echo "Clean complete."
