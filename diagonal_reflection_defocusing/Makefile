
# Makefile for Clawpack code in this directory.
# This version only sets the local files and frequently changed
# options, and then includes the standard makefile pointed to by CLAWMAKE.
CLAWMAKE = $(CLAW)/clawutil/src/Makefile.common

# See the above file for details and a list of make options, or type
#   make .help
# at the unix prompt.


# Adjust these variables if desired:
# ----------------------------------

CLAW_PKG = amrclaw                  # Clawpack package to use
EXE = xamr                          # Executable to create
SETRUN_FILE = setrun.py             # File containing function to make data
OUTDIR = _output                   # Directory for output
SETPLOT_FILE = setplot.py           # File containing function to set plots
PLOTDIR = _plots                    # Directory for plots

OVERWRITE ?= True                   # False ==> make a copy of OUTDIR first

# Check if NLSH_ND_SRC is set, if not, use a relative path
NLSH_ND_SRC ?= ../src

# Environment variable FC should be set to fortran compiler, e.g. gfortran

# Compiler flags can be specified here or set as an environment variable
FFLAGS ?=  -O2 -fopenmp

# ---------------------------------
# package sources for this program:
# ---------------------------------

AMRLIB = $(CLAW)/amrclaw/src/2d
include $(AMRLIB)/Makefile.amr_2d

# ---------------------------------------
# package sources specifically to exclude
# (i.e. if a custom replacement source 
#  under a different name is provided)
# ---------------------------------------

EXCLUDE_MODULES = \

EXCLUDE_SOURCES = \

# ----------------------------------------
# List of custom sources for this program:
# ----------------------------------------

MODULES = \
	mapc2p.f90 \
	$(NLSH_ND_SRC)/2d/soliton_module.f90 \
	$(NLSH_ND_SRC)/2d/cparam_mod.f90 \

SOURCES = \
	qinit.f90 \
	setaux.f90 \
	bc2amr.f90 \
	$(NLSH_ND_SRC)/2d/src2.f90 \
	$(NLSH_ND_SRC)/2d/setprob.f90 \
	$(NLSH_ND_SRC)/2d/rpn2_nlsh_mapped.f90 \
	$(NLSH_ND_SRC)/2d/rpt2_nlsh_mapped.f90 \
	flag2refine2.f90 \
	
#-------------------------------------------------------------------
# Include Makefile containing standard definitions and make options:
include $(CLAWMAKE)

cleanall:
	$(MAKE) clean
	rm -f *.o *.mod *.data *.log


# Makefile for building f2py module from mapc2p.f90
PYVER := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
TARGET := map_module_f2py$(PYVER)

$(TARGET): mapc2p.f90
	f2py -c mapc2p.f90 -m map_module_f2py

mapping: $(TARGET)