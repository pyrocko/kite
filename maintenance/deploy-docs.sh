#!/bin/bash
set -e
VERSION=v`python3 -c "import kite; print(kite.__version__);"`

if [ ! -f maintenance/deploy-docs.sh ] ; then
    echo "must be run from pyrocko's toplevel directory"
    exit 1
fi

cd docs
rm -rf build/$VERSION
make clean; make html $1
cp -r build/html build/$VERSION

read -r -p "Are your sure to update live docs at http://pyrocko.org/kite/docs/$VERSION [y/N]?" resp
case $resp in
    [yY][eE][sS]|[yY] )
        rsync -av build/$VERSION pyrocko@hive:/var/www/pyrocko.org/kite/docs/;
        ;;
    * ) ;;
esac

read -r -p "Do you want to link 'current' to the just uploaded version $VERSION [y/N]?" resp
case $resp in
    [yY][eE][sS]|[yY] )
        echo "Linking /kite/docs/$VERSION to /kite/docs/current";
        ssh pyrocko@hive "rm -rf /var/www/pyrocko.org/kite/docs/current; ln -s /var/www/pyrocko.org/kite/docs/$VERSION /var/www/pyrocko.org/kite/docs/current";
        ;;
    * ) ;;
esac

cd ..
