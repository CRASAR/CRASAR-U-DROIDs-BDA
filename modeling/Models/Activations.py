from torch import nn

#This mapping contains the name of the activation function, and the number of channels that it needs to be 
#expanded to in order for the output channels to map correctly to the perscribed hyperparameters.
ACTIVATION_FUNCTION_MAP = {
	"gelu": [nn.GELU, 1, {}],
	"relu": [nn.ReLU, 1, {}],
	"selu": [nn.SELU, 1, {}],
	"silu": [nn.SiLU, 1, {}],
	"elu": [nn.ELU, 1, {}],
	"glu": [nn.GLU, 2, {"dim": 1}],
}

def getActivationFunction(name):
	return ACTIVATION_FUNCTION_MAP[name.lower().strip()][0]
def getActivationChannels(name):
	return ACTIVATION_FUNCTION_MAP[name.lower().strip()][1]
def getActivationKwargs(name):
	return ACTIVATION_FUNCTION_MAP[name.lower().strip()][2]