#!/bin/bash

source common_configs.sh

echo -e "\n=========================================================================="
echo -e "My environment variables:"
echo -e "--------------------------------------------------------------------------"
printenv
echo -e "==========================================================================\n"

"$PYTHON_COMMAND" "$MODELS_DIR/official/resnet/test_mpi/test_mpi.py"

