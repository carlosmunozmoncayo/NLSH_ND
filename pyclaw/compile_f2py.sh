#! /bin/bash

NLSH_PYCLAW=$(PWD)/$(dirname $BASH_SOURCE)

#Make dir modules if it doesn't exist
if [ ! -d "$NLSH_PYCLAW/modules" ]; then
    mkdir $NLSH_PYCLAW/modules
fi

cd $NLSH_PYCLAW/src/
echo "In $PWD, compiling f2py modules..."

#Compile 1d code
f2py -c rp1_nlsh.f90 -m rs_nlsh_1d
mv rs_nlsh_1d*.so $NLSH_PYCLAW/modules/

#Compile 2d code
f2py -c rpn2_nlsh.f90 rpt2_nlsh.f90 -m rs_nlsh_2d
mv rs_nlsh_2d*.so $NLSH_PYCLAW/modules/

f2py -c rpn2_nlsh_mapped.f90 rpt2_nlsh_mapped.f90 -m rs_nlsh_mapped_2d
mv rs_nlsh_mapped_2d*.so $NLSH_PYCLAW/modules/

cd $NLSH_PYCLAW