import torch.nn as nn
import torch.nn.functional as F
import torch


# Define the ConvNet model used for building the target model. The same structure is used for the shadow model as well
# based on the first attack of the paper's section III.


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3,8,(3,3),1,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2,0)
        )
        
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(8*16*16,200),
            torch.nn.ReLU(),
        )
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(200,10),
            torch.nn.ReLU(),
            torch.nn.LogSoftmax(dim =1)
        )
        
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.linear1(x)
        output = self.linear2(x)
        return output


class MlleaksMLP(torch.nn.Module):
    """
    This is a simple multilayer perceptron with 64-unit hidden layer and a softmax output layer.
    """
    def __init__(self, input_size=3, hidden_size=64, output=1):
        super(MlleaksMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        output = self.fc2(hidden)
        output = self.sigmoid(output)
        return output

class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.features = self._make_layers([64, 'M', 64, 'M', 64, 'M', 64, 'M'])
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)