import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Embedder( nn.Module ):

    def __init__( self, vocab_size, embed_size ):
        super( Embedder, self ).__init__()
        self.vocab_size = vocab_size 
        self.embed_size = embed_size
        self.embeddings = nn.Embedding( vocab_size, embed_size, padding_idx = 0 )
        self.generator = nn.Linear( self.embed_size, self.embed_size, bias = False )

    def forward( self, ids ):
        output = self.embeddings( ids )
        output = F.relu( output )
        output = self.generator( output )
        return output