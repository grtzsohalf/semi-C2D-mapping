import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils import weight_init


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# The WGAN-GP module
# ==================
#

LAMBDA = 1.
USE_CUDA = True


######################################################################
# The generator module
# --------------------
#

class Generator(nn.Module):

    def __init__(self, ):
        self.null = None

    def forward(self, inputs):
        return 


######################################################################
# The discriminator module
# ------------------------
#

class FCDiscriminator(nn.Module):
    """
    Args:
        input_size:
        hidden_size:
        n_layers:
    Inputs: inputs
        - **inputs** (batch_size, dimensions): tensor to be discriminated
    Outputs: output
        - **output** (batch_size): real number indicating reality
    """

    def __init__(self, input_size, hidden_size, n_layers):
        super(FCDiscriminator, self).__init__()

        self.fc_tensor = self._makeFcSequential(input_size, hidden_size, n_layers)
        self.fc_tensor.apply(weight_init)

    def _makeFcSequential(self, input_size, hidden_size, n_layers):
        fc_list = [nn.Linear(input_size, hidden_size), nn.LeakyReLU()]
        for i in range(n_layers-2):
            fc_list.append(nn.Linear(hidden_size, hidden_size))
            fc_list.append(nn.LeakyReLU())
        fc_list.append(nn.Linear(hidden_size, 1))
        return nn.Sequential(*fc_list)

    def forward(self, inputs):
        return self.fc_tensor(inputs)

    
######################################################################
# The gradient penalty function
# -----------------------------
#

def calc_gradient_penalty(batch_size, netD, real_data, fake_data):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if USE_CUDA else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if USE_CUDA:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if USE_CUDA else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
