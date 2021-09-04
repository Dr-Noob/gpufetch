CXX ?= g++
CUDA_PATH ?= /usr/local/cuda/
PREFIX ?= /usr

CXXFLAGS+=-Wall -Wextra -pedantic -fstack-protector-all -pedantic
SANITY_FLAGS=-Wfloat-equal -Wshadow -Wpointer-arith

SRC_COMMON=src/common/
SRC_CUDA=src/cuda/

COMMON_SRC = $(SRC_COMMON)main.cpp  $(SRC_COMMON)gpu.cpp $(SRC_COMMON)args.cpp $(SRC_COMMON)global.cpp $(SRC_COMMON)printer.cpp $(SRC_COMMON)pci.cpp
COMMON_HDR = $(SRC_COMMON)ascii.hpp $(SRC_COMMON)gpu.hpp $(SRC_COMMON)args.hpp $(SRC_COMMON)global.hpp $(SRC_COMMON)printer.hpp $(SRC_COMMON)pci.hpp

CUDA_SRC = $(SRC_CUDA)cuda.cpp $(SRC_CUDA)uarch.cpp $(SRC_CUDA)pci.cpp
CUDA_HDR = $(SRC_CUDA)cuda.hpp $(SRC_CUDA)uarch.hpp $(SRC_CUDA)pci.hpp $(SRC_CUDA)chips.hpp

SOURCE += $(COMMON_SRC) $(CUDA_SRC)
HEADERS += $(COMMON_HDR) $(CUDA_HDR)

OUTPUT=gpufetch

CXXFLAGS+= -I pciutils/install/include -I $(CUDA_PATH)/samples/common/inc -I $(CUDA_PATH)/targets/x86_64-linux/include -L $(CUDA_PATH)/targets/x86_64-linux/lib -L pciutils/install/lib
LDFLAGS+=-lcudart -lpci

all: CXXFLAGS += -O3
all: $(OUTPUT)

debug: CXXFLAGS += -g -O0
debug: $(OUTPUT)

static: CXXFLAGS += -static -O3
static: $(OUTPUT)

strict: CXXFLAGS += -O3 -Werror -fsanitize=undefined -D_FORTIFY_SOURCE=2
strict: $(OUTPUT)

$(OUTPUT): Makefile $(SOURCE) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SANITY_FLAGS) $(SOURCE) $(LDFLAGS) -o $(OUTPUT)

run: $(OUTPUT)
	./$(OUTPUT)

clean:
	@rm -f $(OUTPUT)

install: $(OUTPUT)
	install -Dm755 "gpufetch"   "$(DESTDIR)$(PREFIX)/bin/gpufetch"
	install -Dm644 "LICENSE"    "$(DESTDIR)$(PREFIX)/share/licenses/gpufetch-git/LICENSE"
	install -Dm644 "gpufetch.1" "$(DESTDIR)$(PREFIX)/share/man/man1/gpufetch.1.gz"

uninstall:
	rm -f "$(DESTDIR)$(PREFIX)/bin/gpufetch"
	rm -f "$(DESTDIR)$(PREFIX)/share/licenses/gpufetch-git/LICENSE"
	rm -f "$(DESTDIR)$(PREFIX)/share/man/man1/gpufetch.1.gz"
