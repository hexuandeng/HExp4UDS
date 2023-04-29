import torch
from heuds.modules.embeddings.base_embedding import BaseEmbedding


class TokenCharactersEncoder(BaseEmbedding):
    """
    A ``TokenCharactersEncoder`` takes the output of a
    :class:`~allennlp.data.token_indexers.TokenCharactersIndexer`, which is a tensor of shape
    (batch_size, num_tokens, num_characters), embeds the characters, runs a token-level encoder, and
    returns the result, which is a tensor of shape (batch_size, num_tokens, encoding_dim).  We also
    optionally apply dropout after the token-level encoder.

    We take the embedding and encoding modules as input, so this class is itself quite simple.
    """
    def __init__(self, embedding, encoder, dropout: float = 0.0) -> None:
        super(TokenCharactersEncoder, self).__init__()
        self._embedding = embedding
        self._encoder = encoder
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

    def get_output_dim(self) -> int:
        return self._encoder._module.get_output_dim()  # pylint: disable=protected-access

    def forward(self, token_characters: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        mask = (token_characters != 0).long()
        return self._dropout(self._encoder(self._embedding(token_characters), mask))
