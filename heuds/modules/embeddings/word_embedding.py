import torch
from torch.nn.functional import embedding
from dataclasses import dataclass, field
from heuds.base.base_vocab import BaseVocab
from heuds.base.base_config import BaseConfig
from heuds.modules.embeddings.base_embedding import BaseEmbedding


@dataclass
class WordEmbeddingConfig(BaseConfig):
    embed_dim: int = field(
        default=128, metadata={"help": "embedding dimension"}
    )
    proj_dim: int = field(
        default=None, metadata={"help": "Whether using projection layer and its corresponding dimension."}
    )
    trainable: bool = field(
        default=True, metadata={"help": "Whether freeze the embedding weight."}
    )
    max_norm: float = field(
        default=None, metadata={"help": "If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm. Note: this will modify weight in-place."}
    )
    norm_type: float = field(
        default=2., metadata={"help": "The p of the p-norm to compute for the max_norm option."}
    )
    scale_grad_by_freq: bool = field(
        default=False, metadata={"help": "If given, this will scale gradients by the inverse of frequency of the words in the mini-batch."}
    )
    sparse: bool = field(
        default=False, metadata={"help": "If True, gradient w.r.t. weight will be a sparse tensor."}
    )


class WordEmbedding(BaseEmbedding):
    """
    A more featureful embedding module than the default in Pytorch.  Adds the ability to:

        1. embed higher-order inputs
        2. pre-specify the weight matrix
        3. use a non-trainable embedding

    Parameters
    ----------
    voc_size : int
        Size of the dictionary of embeddings (vocabulary size).
    embed_dim : int
        The size of each embedding vector.
    proj_dim : int, (optional, default=None)
        If given, we add a projection layer after the embedding layer.  This really only makes
        sense if ``trainable`` is ``False``.
    weight : torch.FloatTensor, (optional, default=None)
        A pre-initialised weight matrix for the embedding lookup, allowing the use of
        pretrained vectors.
    padding_index : int, (optional, default=None)
        If given, pads the output with zeros whenever it encounters the index.
    trainable : bool, (optional, default=True)
        Whether or not to optimize the embedding parameters.
    max_norm : float, (optional, default=None)
        If given, will renormalize the embeddings to always have a norm lesser than this
    norm_type : float, (optional, default=2)
        The p of the p-norm to compute for the max_norm option
    scale_grad_by_freq : boolean, (optional, default=False)
        If given, this will scale gradients by the frequency of the words in the mini-batch.
    sparse : bool, (optional, default=False)
        Whether or not the Pytorch backend should use a sparse representation of the embedding weight.
    vocab_namespace : str, (optional, default=None)
        In case of fine-tuning/transfer learning, the model's embedding matrix needs to be
        extended according to the size of extended-vocabulary. To be able to know how much to
        extend the embedding-matrix, it's necessary to know which vocab_namspace was used to
        construct it in the original training. We store vocab_namespace used during the original
        training as an attribute, so that it can be retrieved during fine-tuning.
    pretrained_file : str, (optional, default=None)
        Used to keep track of what is the source of the weights and loading more embeddings at test time.
        **It does not load the weights from this pretrained_file.** For that purpose, use
        ``Embedding.from_params``.

    Returns
    -------
    An Embedding module.
    """

    def __init__(self, cfg,
                 voc_size: int = None,
                 weight: torch.FloatTensor = None,
                 padding_index: int = 0) -> None:
        super(WordEmbedding, self).__init__()
        self.cfg = cfg
        self.voc_size = voc_size
        self.padding_index = padding_index

        if weight is None:
            weight = torch.FloatTensor(voc_size, cfg.embed_dim)
            self.weight = torch.nn.Parameter(
                weight, requires_grad=cfg.trainable)
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            if weight.size() != (voc_size, cfg.embed_dim):
                raise ValueError(
                    "A weight matrix was passed with contradictory embedding shapes.")
            self.weight = torch.nn.Parameter(
                weight, requires_grad=cfg.trainable)

        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

        if cfg.proj_dim:
            self._projection = torch.nn.Linear(cfg.embed_dim, cfg.proj_dim)
        else:
            self._projection = None

    @property
    def output_dim(self):
        return self.cfg.proj_dim or self.cfg.embed_dim

    # Custom logic requires custom from_params.
    @classmethod
    def from_vocab(cls, cfg, vocab: BaseVocab, padding_index: int = 0) -> 'Embedding':  # type: ignore
        """
        We need the vocabulary here to know how many items we need to embed, and we look for a
        ``vocab_namespace`` key in the parameter dictionary to know which vocabulary to use.  If
        you know beforehand exactly how many embeddings you need, or aren't using a vocabulary
        mapping for the things getting embedded here, then you can pass in the ``voc_size``
        key directly, and the vocabulary will be ignored.

        In the configuration file, a file containing pretrained embeddings can be specified
        using the parameter ``"pretrained_file"``.
        It can be the path to a local file or an URL of a (cached) remote file.
        Two formats are supported:

            * hdf5 file - containing an embedding matrix in the form of a torch.Tensor;

            * text file - an utf-8 encoded text file with space separated fields::

                    [word] [dim 1] [dim 2] ...

              The text file can eventually be compressed with gzip, bz2, lzma or zip.
              You can even select a single file inside an archive containing multiple files
              using the URI::

                    "(archive_uri)#file_path_inside_the_archive"

              where ``archive_uri`` can be a file system path or a URL. For example::

                    "(https://nlp.stanford.edu/data/glove.twitter.27B.zip)#glove.twitter.27B.200d.txt"
        """

        # pylint: disable=arguments-differ
        voc_size = vocab.vocab_size
        if hasattr(vocab, 'emb_matrix'):
            cfg.embed_dim = vocab.emb_size
            return cls(cfg,
                       voc_size=voc_size,
                       weight=torch.FloatTensor(vocab.emb_matrix),
                       padding_index=padding_index)
        else:
            return cls(cfg,
                       voc_size=voc_size,
                       weight=None,
                       padding_index=padding_index)

    def forward(self, inputs):
        inputs = inputs.to(torch.int)

        embedded = embedding(inputs, self.weight,
                             padding_idx=self.padding_index,
                             max_norm=self.cfg.max_norm,
                             norm_type=self.cfg.norm_type,
                             scale_grad_by_freq=self.cfg.scale_grad_by_freq,
                             sparse=self.cfg.sparse)

        if self._projection:
            embedded = self._projection(embedded)
        return embedded
