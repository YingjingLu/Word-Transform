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

        self.src_optimizer = torch.optim.Adam( self.src_embed.params, lr=5e-5 )
        self.disc_optimizer = torch.optim.Adam( self.disc.params, lr=5e-5 )

    def train( self, src_sents, tar_sents ):
        src_ids = self.src_vocab.sentences2ids( src_sents )
        src_tensor = torch.tensor(src_ids ).view( -1, 1 )

        tar_ids = self.tar_vocab.sentences2ids( tar_sents )
        tar_tensor = torch.tensor(tar_ids ).view( -1, 1 )

        src_label = torch.zeros( src_tensor.size() )
        tar_label = torch.ones( tar_tensor.size() )

        self.src_optimizer.zero_grad()
        loss = self.loss( self.disc( self.src_embed( src_tensor ) ), tar_label )
        loss.backward()
        self.src_optimizer.step()

        self.disc_optimizer.zero_grad()
        loss = self.loss( self.disc( self.tar_embed( tar_tensor ) ), tar_label ) + \
               self.loss( self.disc( self.src_embed( src_tensor ) ), src_label )
        loss.backward()
        self.disc_optimizer.step()

    def tranform( self, src_sents ):
        src_ids = self.src_vocab.sentences2ids( src_sents )
        src_tensor = torch.tensor(src_ids ).view( -1, 1 )

        return self.src_embed( src_tensor )


