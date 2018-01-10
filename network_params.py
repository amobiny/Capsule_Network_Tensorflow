import tensorflow as tf

# Parameters of Conv1_layer
conv1_params = {"filters": 256,
                "kernel_size": 9,
                "strides": 1,
                "padding": "valid",
                "activation": tf.nn.relu}

# Parameters of PrimaryCaps_layer
caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules
caps1_n_dims = 8

conv2_params = {"filters": caps1_n_maps * caps1_n_dims,  # 256 convolutional filters
                "kernel_size": 9,
                "strides": 2,
                "padding": "valid",
                "activation": tf.nn.relu}

# Parameters of DigitCaps_layer
caps2_n_caps = 10
caps2_n_dims = 16

# Parameters of the Decoder
n_hidden1 = 512
n_hidden2 = 1024
n_output = 28 * 28
