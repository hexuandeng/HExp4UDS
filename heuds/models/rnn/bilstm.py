import torch
from torch import nn
from dataclasses import dataclass, field
from heuds.config.base_config import BaseConfig
from heuds.models.fairseq_incremental_decoder import FairseqIncrementalDecoder

@dataclass
class LstmEncoderConfig(BaseConfig):
    hidden_size: int = field(
        default=512, metadata={"help": "hidden_size"}
    )
    num_layers: int = field(
        default=2, metadata={"help": "num_layers"}
    )
    dropout: float = field(
        default=0.3, metadata={"help": "dropout"}
    )


@dataclass
class LstmDecoderConfig(LstmEncoderConfig):
    hidden_size: int = field(
        default=1024, metadata={"help": "hidden_size"}
    )


class LstmEncoder(nn.Module):
    def __init__(self, cfg, input_embed_dim) -> None:
        super().__init__()
        self.model = nn.LSTM(input_embed_dim, cfg.hidden_size, cfg.num_layers,
                                   dropout=cfg.dropout, bidirectional=True, batch_first=True)
        self.output_dim = cfg.hidden_size * 2

    def forward(self, token_embeddings, encoder_mask=None, initial_state=None):
        output, (h_n, c_n) = self.model(token_embeddings, initial_state)
        assert h_n.shape[0] % 2 == 0 and c_n.shape[0] % 2 == 0
        h_n = h_n.transpose(0, 1).contiguous().view(h_n.shape[1], int(
            h_n.shape[0] / 2), -1)
        c_n = c_n.transpose(0, 1).contiguous().view(c_n.shape[1], int(
            c_n.shape[0] / 2), -1)
        return {
            "encoder_out": output,
            "hidden_state": (h_n, c_n)
        }


class LstmDecoder(FairseqIncrementalDecoder):
    def __init__(self, cfg, input_embed_dim) -> None:
        super().__init__()
        self.model = nn.LSTM(input_embed_dim, cfg.hidden_size, cfg.num_layers,
                                   dropout=cfg.dropout, bidirectional=False, batch_first=True)
        self.output_dim = cfg.hidden_size

    def forward(self, prev_output_embeddings, decoder_mask=None, encoder_out=None, incremental_state=None):
        flag = True
        if incremental_state is not None:
            prev_output_embeddings = prev_output_embeddings[:, -1:]
            prev_decoder_out = self.get_incremental_state(incremental_state, "decoder_out")
            prev_hidden_state = self.get_incremental_state(incremental_state, "hidden_state")
            if prev_decoder_out is not None and prev_hidden_state is not None:
                tmp = []
                for item in prev_hidden_state:
                    tmp.append(item.transpose(0, 1).contiguous())
                output, (h_n, c_n) = self.model(
                    prev_output_embeddings, type(prev_hidden_state)(tmp))
                output = torch.cat((prev_decoder_out, output), dim=1)
                flag = False
        if flag:
            if encoder_out is None:
                output, (h_n, c_n) = self.model(prev_output_embeddings)
            else:
                hidden_state = []
                for item in encoder_out["hidden_state"]:
                    hidden_state.append(item.transpose(0, 1).contiguous())
                output, (h_n, c_n) = self.model(
                    prev_output_embeddings, type(encoder_out["hidden_state"])(hidden_state))
            
        h_n = h_n.transpose(0, 1).contiguous()
        c_n = c_n.transpose(0, 1).contiguous()

        if incremental_state is not None:
            self.set_incremental_state(incremental_state, "decoder_out", output)
            self.set_incremental_state(incremental_state, "hidden_state", (h_n, c_n))

        return {
            "decoder_out": output,
            "hidden_state": (h_n, c_n)
        }

    def reorder_incremental_state(
        self,
        incremental_state,
        new_order
    ):
        return super().reorder_incremental_state(incremental_state, new_order, ["decoder_out", "hidden_state"])
