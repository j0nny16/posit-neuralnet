#! /bin/bash -l

if [ ! -d $1 ]
then
    echo "usage: $0 <project>"
    exit 1
fi

cd $1
cmake -Bbuild .
cmake --build build --config Release -j8
