
# Installation

## Prerequisites
```
Python 3.6+
numpy
opencv-contrib-python
```

## Install
```
pip3 install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Run
```
1. 영상 확인
python3 bin/ar_markers_scan.py -c config.json

2. 영상 저장 (-r)
python3 bin/ar_markers_scan.py -c config.json -r

```
