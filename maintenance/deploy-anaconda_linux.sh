#!/bin/bash
if [ ! -f deploy-anaconda_linux.sh ] ; then
    echo "must be run from kite's maintenance directory"
    exit 1
fi

branch="$1"
if [ -z "$branch" ]; then
    branch=master
fi

echo "Building pyrocko-kite for Anaconda on branch $branch"
rm -rf "anaconda/kite.git"
git clone -b $branch "../" "anaconda/kite.git"
rm -rf "anaconda/kite.git/maintenance/anaconda/meta.yaml"

anaconda/deploy.sh $1
