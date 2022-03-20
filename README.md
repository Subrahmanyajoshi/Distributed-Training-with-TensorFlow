# Distributed Training with TensorFlow

- This repository shows how to distribute training of large Tensorflow models to make it faster.
- In tensorflow training can be distributed using strategies module.
- Here are the results of my experiments on different strategies. I had to experiment with only one GPU, as no matter 
how much I tried, I was not able to get multiple GPUs to see how it affects training time.

## Note
- All the experiments below were run on a Google Colab notebook with 26 GB RAM.
- Amazon reviews dataset obtained from [kaggle](https://www.kaggle.com/bittlingmayer/amazonreviews) was used.
- A reduced version of original dataset was used after some preprocessing. 
- Train dataset size was 1,000,000 (1 million). Validation dataset size was 360,000 (360 thousand).

##### By default, the implementation uses Mirrored Strategy. If running on TPU, select TPU strategy [here](https://github.com/Subrahmanyajoshi/Distributed-Training-with-TensorFlow/blob/main/trainer/task.py#L114)

## 1. Mirrored Strategy without any GPUs.
 - Batch size: 512.
 - Time taken: 6171 seconds per epoch (102.85 minutes).
 - Validation Metrics at the end of first epoch: val_loss: 0.2168 - val_accuracy: 0.9129.
 - Couldn't try higher batch sizes as CPU usage was above 90%.
 
## 2. Mirrored Strategy with 1 NVIDIA Tesla P100-PCIe GPU
### Experiment 1
- Batch size: 512.
- Time taken: 1321 seconds per epoch (22.01 minutes).
- Validation Metrics at the end of first epoch: val_loss: 0.2137 - val_accuracy: 0.9138.
- GPU utilization was less than 26%. So I could increase batch size.

### Experiment 2
- Batch size: 2056.
- Time taken: 390 seconds per epoch (6.5 minutes).
- Validation Metrics at the end of first epoch: val_loss: 0.2251 - val_accuracy: 0.9092.
- GPU utilization was more than 80%.


## 3. TPU Strategy with 1 TPU.
### Experiment 1
- Batch size: 512.
- A TPU system contains 8 TPUs. Hence, the final batch size is multiplied by 8.
- Time taken: 106 seconds per epoch (1.7 minutes).
- Validation Metrics at the end of first epoch: val_loss: 0.2345 - val_accuracy: 0.9049.
- TPU utilization was 2%. So I could try a higher batch size.

### Experiment 2
- Batch size: 2056.
- A TPU system contains 8 TPUs. Hence, the final batch size is multiplied by 8.
- Time taken: 100 seconds per epoch (1.6 minutes).
- Validation Metrics at the end of first epoch: val_loss: 0.2823 - val_accuracy: 0.8856.
- TPU utilization was still 2.2%. The utilization might increase if the network is deeper. I encountered the following 
error when I tried to increase the batch size further.
```text
tensorflow.python.framework.errors_impl.ResourceExhaustedError: received trailing metadata size exceeds limit
```



