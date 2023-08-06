from torch import nn


class WMMapper(nn.Module):
    def __init__(self, bitlength: int):
        super().__init__()
        self.bitlength = bitlength
        self.msg = None

    def get_msg(self):
        return self.msg

    def set_msg(self, msg):
        self.msg = msg