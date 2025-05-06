class Layer:

def update_params(self, learning_rate):
    pass

class Dense(Layer):
    def forward(self, x):
        self.x = x
        return x @ self.weights + self.bias
    def backward(self, dout):
        self.dweights = self.x.T @ dout
        self.dbias = np.sum(dout, axis=0)
        return dout @ self.weights.T
    def update_params(self, learning_rate):
        self.weights -= learning_rate * self.dweights
        self.bias -= learning_rate * self.dbias


class Flatten(Layer):
    pass

class Conv2D(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        

    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, x):
        raise NotImplementedError


    
class MaxPool2D(Layer):
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
    
    def backward(self, dout):
        dx = np.zeros_like(self.x)
        batch_size, channels, out_height, out_width = dout.shape

        for n in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        dx[n, c, h_start:h_end, w_start:w_end] += (self.max_mask[n, c, h_start:h_end, w_start:w_end] * dout[n, c, i, j])

        return dx
    
    