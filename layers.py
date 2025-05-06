import numpy as np

class Layer:
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, x):
        raise NotImplementedError
    
    def update_params(self, learning_rate):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        limit = np.sqrt(1 / input_size)
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros((1, output_size))

    def forward(self, x):
        self.x = x
        return x @ self.weights + self.bias
    
    def backward(self, d_out):
        self.dW = self.x.T @ d_out
        self.db = np.sum(d_out, axis=0, keepdims=True)
        return d_out @ self.weights.T
    
    def update_params(self, learning_rate):
        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.db

class Flatten(Layer):
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(self.input_shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        limit = np.sqrt(1 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros((out_channels, 1))

    def forward(self, x):
        self.x = x
        batch_size, channels, in_h, in_w = x.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        # calculating the output size
        out_h = (in_h - k + 2 * p) // s + 1
        out_w = (in_w - k + 2 * p) // s + 1

        self.output = np.zeros((batch_size, self.out_channels, out_h, out_w))

        # padding
        if p > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        else:
            x_padded = x

        # applying the filter to input
        for n in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * s
                        h_end = h_start + k
                        w_start = j * s
                        w_end = w_start + k

                        region = x_padded[n, :, h_start:h_end, w_start:w_end]
                        self.output[n, oc, i, j] = np.sum(region * self.weights[oc]) + self.bias[oc]

        return self.output

    # d_out must be the gradient from the next layer
    # d_out must have the same shape as self.output
    def backward(self, d_out):
        x = self.x
        k = self.kernel_size
        s = self.stride
        p = self.padding
        batch_size, channels, in_h, in_w = x.shape
        _, _, out_h, out_w = d_out.shape

        # padding
        if p > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
            dx_padded = np.zeros_like(x_padded)
        else:
            x_padded = x
            dx_padded = np.zeros_like(x)

        # initializing gradients
        dW = np.zeros_like(self.weights)
        db = np.zeros_like(self.bias)

        # gradient loop
        for n in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * s
                        h_end = h_start + k
                        w_start = j * s
                        w_end = w_start + k

                        region = x_padded[n, :, h_start:h_end, w_start:w_end]
                        dW[oc] += d_out[n, oc, i, j] * region
                        db[oc] += d_out[n, oc, i, j]
                        dx_padded[n, :, h_start:h_end, w_start:w_end] += d_out[n, oc, i, j] * self.weights[oc]

        self.dW = dW
        self.db = db

        # remove padding if needed
        if p > 0:
            return dx_padded[:, :, p:-p, p:-p]
        else:
            return dx_padded

    def update_params(self, learning_rate):
        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.db
    
class MaxPool2D(Layer):
    def __init__(self, kernel_size = 2, stride = 2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        self.x = x
        batch_size, channels, height, width = x.shape
        out_height = (height - self.kernel_size)// self.stride + 1
        out_width = (width - self.kernel_size)//self.stride +1
        out = np.zeros((batch_size, channels, out_height, out_width))
        self.max_mask = np.zeros_like(x)
        
        for n in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        region = x[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(region)
                        out[n, c, i, j] = max_val

                        mask = (region == max_val)
                        self.max_mask[n, c, h_start:h_end, w_start:w_end] += mask
        return out
    
    def backward(self, d_out):
        dx = np.zeros_like(self.x)
        batch_size, channels, out_height, out_width = d_out.shape

        for n in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        dx[n, c, h_start:h_end, w_start:w_end] += (self.max_mask[n, c, h_start:h_end, w_start:w_end] * d_out[n, c, i, j])

        return dx
    
    