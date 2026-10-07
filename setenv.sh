#!/bin/bash

#Set some environment variables for the NLS_ND project
#First check if $CLAW is set
if [ -z "$CLAW" ]; then
    echo "Error: \$CLAW is not set. Please set it to the root directory of your clawpack installation."
    exit 1
fi

NLSH_ND_SRC=$(PWD)/$(dirname $BASH_SOURCE)
NLSH_ND_SRC="${NLSH_ND_SRC%/*}/src"
export NLSH_ND_SRC="$NLSH_ND_SRC"
echo "NLSH_ND_SRC is set to $NLSH_ND_SRC"