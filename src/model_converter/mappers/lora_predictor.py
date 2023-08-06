import torch.nn as nn


class LoRAPredictor(nn.Module):
    def __init__(self, message_size, u_shape, v_shape):
        super(LoRAPredictor, self).__init__()

        self.u_shape = u_shape
        self.v_shape = v_shape

        # The output size is the sum of the number of parameters in U and V
        output_size = u_shape[0] * u_shape[1] + v_shape[0] * v_shape[1]

        # Define the dense layer
        # You can extend this to multiple layers or change activation functions if needed
        self.dense = nn.Sequential(
            nn.Linear(message_size, output_size),
            nn.Tanh()  # Activation function (can be changed)
        )

    def forward(self, x):
        # Pass through the dense layer
        out = self.dense(x)

        # Split the output into two parts for U and V and reshape them
        u_elems = self.u_shape[0] * self.u_shape[1]
        u, v_flattened = out.split(u_elems, dim=1)
        u = u.reshape(-1, *self.u_shape)
        v = v_flattened.reshape(-1, *self.v_shape)

        return u, v