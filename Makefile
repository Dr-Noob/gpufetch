CXX ?= g++

CXXFLAGS+=-Wall -Wextra -pedantic -fstack-protector-all -pedantic
SANITY_FLAGS=-Wfloat-equal -Wshadow -Wpointer-arith

PREFIX ?= /usr

SRC_COMMON=src/common/

COMMON_SRC = $(SRC_COMMON)main.c $(SRC_COMMON)args.c $(SRC_COMMON)global.c
COMMON_HDR = $(SRC_COMMON)args.h $(SRC_COMMON)global.h

SOURCE += $(COMMON_SRC)
HEADERS += $(COMMON_HDR)
OUTPUT=gpufetch

all: CXXFLAGS += -O3
all: $(OUTPUT)

debug: CXXFLAGS += -g -O0
debug: $(OUTPUT)

static: CXXFLAGS += -static -O3
static: $(OUTPUT)

strict: CXXFLAGS += -O3 -Werror -fsanitize=undefined -D_FORTIFY_SOURCE=2
strict: $(OUTPUT)

$(OUTPUT): Makefile $(SOURCE) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SANITY_FLAGS) $(SOURCE) -o $(OUTPUT)

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
