# Default values.
CXX ?= g++
AR = gcc-ar
SSE_FLAGS = -march=native
CXXFLAGS ?= -Wall -Wextra -Wundef -pedantic -g -O3 -march=native -fno-math-errno -funsafe-math-optimizations -flto -pipe -fopt-info-optimized-vec

# Delete this if libgomp.a is not installed! (e.g., OS X without gcc)
OPENMP ?= -fopenmp

# Always compile programs and libs with multi-threading.
CXXFLAGS += -pthread

UNAME = $(shell uname)
ifeq (${UNAME},Darwin)
    # Use a recent OS X SDK - homebrew qt5 needs it.
    CXXFLAGS += -mmacosx-version-min=10.6
    LDFLAGS += -mmacosx-version-min=10.6
endif

LDFLAGS += ${CXXFLAGS}

COMPILE.cc = ${CXX} ${CXXFLAGS} ${CPPFLAGS} -c
LINK.o = ${CXX} ${LDFLAGS}
.DEFAULT_GOAL := ${TARGET}

# How to link example from example.o and other files.
%: %.o
	${LINK.o} -o $@ $^ ${LDLIBS}

# How to compile example.o from example.cc.
%.o: %.cc
	${COMPILE.cc} -o $@ $<

# Library paths.
vpath libmve_dmrecon.a ${MVE_ROOT}/libs/dmrecon/
vpath libmve_math.a ${MVE_ROOT}/libs/math/
vpath libmve.a ${MVE_ROOT}/libs/mve/
vpath libmve_ogl.a ${MVE_ROOT}/libs/ogl/
vpath libmve_util.a ${MVE_ROOT}/libs/util/
vpath libmve_sfm.a ${MVE_ROOT}/libs/sfm/

# Autogenerate dependencies.
Makefile.dep: [^_]*.cc
	${COMPILE.cc} -MM $^ >$@

-include Makefile.dep
