#!/usr/bin/env

echo -e "\n=========================================================================="
echo -e "My environment variables:"
echo -e "--------------------------------------------------------------------------"
printenv
echo -e "==========================================================================\n"

source common_configs.sh
"$PYTHON_COMMAND" mpi_spawn_test.py

