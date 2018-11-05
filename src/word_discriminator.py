import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Discriminator( nn.Module ):

    def __init__( self, input_size, hidden_size ):
        super( Discriminator, self ).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.l0 = nn.Linear( self.input_size, self.hidden_size )
        self.l1 = nn.Linear( self.hidden_size, self.hidden_size )
        self.l2 = nn.Linear( self.hidden_size, self.hidden_size )
        self.l3 = nn.Linear( self.hidden_size, self.hidden_size )
        self.l4 = nn.Linear( self.hidden_size, 1 )

    def forward( self, inputs ):
        inputs = inputs.view( -1, self.input_size )
        inputs = F.relu( self.l0( inputs ) )
        inputs = F.relu( self.l1( inputs ) )
        inputs = F.relu( self.l2( inputs ) )
        inputs = F.relu( self.l3( inputs ) )
        inputs = self.l4( inputs )
        return inputs

