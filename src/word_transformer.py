import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from word_discriminator import * 
from word_embedder import Embedder 

class WordTransformer( object ):

    def __init__( self, src_vocab, tar_vocab, src_vocab_size, tar_vocab_size, embed_size, hidden_size, tar_embed ):
        self.src_vocab = src_vocab 
        self.tar_vocab = tar_vocab 
        self.src_vocab_size = src_vocab_size
        self.tar_vocab_size = tar_vocab_size
        self.embed_size = embed_size 
        self.hidden_size = hidden_size 
        self.tar_embed = tar_embed

        self.src_embed = Embedder( src_vocab_size, embed_size )

        self.disc = Discriminator( embed_size, hidden_size )
        self.loss = nn.CrossEntropyLoss().cuda()

    def train( self, src_sents, tar_sents ):
        pass

    


