#!/usr/bin/env bash

script_name=$(basename $(echo -n $0))

# echo "${script_name}"
echo ""

if [ $# -lt 1 -o $# -gt 2 ] ; then
  echo "Usage ${script_name} dir-path --not-cancel"
  exit -1
fi

flag_cancel_subdirs_tests=1
if [ $# -eq 2 ] ; then
  flag_cancel_subdirs_tests=$2
  if [ "$2" != "--not-cancel" ] ; then
    echo "ERROR: '$2' flag is not allowed."
    echo "Usage ${script_name} dir-path --not-cancel"
    exit -1
  else
    flag_cancel_subdirs_tests=0
  fi
fi

results_tests_dir=$1

if [ ! -d "${results_tests_dir}" ] ; then
  echo "ERROR: path to directory '${results_tests_dir}' does not exists!"
  exit -2
fi

num_tests_dirs=$(find ${results_tests_dir} \
    -maxdepth 1 \
    -mindepth 1 \
    -type d \
    | egrep -E "test$" \
    | wc -l)

if [ $num_tests_dirs -eq 0 ]  ; then
  echo -e "Nothing to do, no tests dirs founded."
  exit 0
fi
echo -e "Founded ${num_tests_dirs} tests dirs to be removed, which are:\n"

tests_dirs=$(find ${results_tests_dir} \
    -maxdepth 1 \
    -mindepth 1 \
    -type d \
    | egrep -E "test$")
echo -n "${tests_dirs}" | awk '{ printf("%s\n", $0) }'

if [ $flag_cancel_subdirs_tests -eq 1 ] ; then
  exit 0
fi

echo -e "\nRemoving tests dirs..."
echo -n "${tests_dirs}" | awk '{ system(sprintf("rm -fr %s/", $0)) }'
echo "Removing tests dirs: Done."




