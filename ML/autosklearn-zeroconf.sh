# A compiler (gcc) is needed to compile a few things the from auto-sklearn requirements.txt
# Chose just the line for your Linux flavor below

# On Ubuntu
sudo apt-get install gcc build-essential swig

# On CentOS 7-1611 http://www.osboxes.org/centos/ https://drive.google.com/file/d/0B_HAFnYs6Ur-bl8wUWZfcHVpMm8/view?usp=sharing
sudo yum -y update 
sudo reboot
sudo yum install epel-release python34 python34-devel python34-setuptools
sudo yum -y groupinstall 'Development Tools'

# auto-sklearn requires swig 3.0 
wget downloads.sourceforge.net/project/swig/swig/swig-3.0.12/swig-3.0.12.tar.gz -O swig-3.0.12.tar.gz
tar xf swig-3.0.12.tar.gz 
cd swig-3.0.12 
./configure --without-pcre
make
sudo make install
cd ..

sudo easy_install-3.4 pip
# if you want to use virtual environments
sudo pip3 install virtualenv
virtualenv zeroconf -p /usr/bin/python3.4
source zeroconf/bin/activate

curl https://raw.githubusercontent.com/paypal/autosklearn-zeroconf/master/requirements.txt | xargs -n 1 -L 1 pip install
