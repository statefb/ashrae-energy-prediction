```
$ git clone https://github.com/statefb/ashrae-energy-prediction
$ sudo apt -y update
$ sudo apt -y upgrade
$ sudo apt -y install python3-pip
$ pip3 install kaggle
$ mkdir .kaggle
$ nano .kaggle/kaggle.json
$ chmod 600 .kaggle/kaggle.json
$ kaggle competitions download -c ashrae-energy-prediction
$ sudo apt install -y unzip
$ unzip ashrae-energy-prediction.zip
$ mv *.csv ashrae-energy-prediction/data/raw

$ nohup python main.py > ~/out.log &
```