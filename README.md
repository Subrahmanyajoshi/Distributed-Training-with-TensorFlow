# Distributed-Training-with-Tensorflow
This repository shows how to distribute training of large machine learning models to make it faster.

1. Mirrored strategy without any GPUs. A colab server with 26 GB ram. batch size 512
1953/1953 [==============================] - 6171s 3s/step - loss: 0.2431 - accuracy: 0.8990 - val_loss: 0.2168 - val_accuracy: 0.9129
-- CPU usage is continously above 90%

2. Mirrored strategy with 1 nvidia tesla p100-pcie GPU, batch size 512
1953/1953 [==============================] - 1321s 670ms/step - loss: 0.2426 - accuracy: 0.8994 - val_loss: 0.2137 - val_accuracy: 0.9138
---------- GPU utilization is less than 26%---------

3. Mirrored strategy with 1 nvidia tesla p100-pcie GPU, batch size 2056
486/486 [==============================] - 390s 790ms/step - loss: 0.2667 - accuracy: 0.8862 - val_loss: 0.2251 - val_accuracy: 0.9092

