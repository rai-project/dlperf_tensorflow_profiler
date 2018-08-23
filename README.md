# dlperf_tensorflow_profiler

## Setup

```
pip install opencv-python
```

## AlexNet Inference


wget https://s3.amazonaws.com/store.carml.org/models/tensorflow/models/bvlc_alexnet_1.0/frozen_model.pb
mv frozen_model.pb alexnet.pb

```
cd inference
python alexnet_infer.py ../_fixtures/920x920_hamburger.jpg
```

### Enable TF Verbose

```
export TF_CPP_MIN_VLOG_LEVEL=3
```
