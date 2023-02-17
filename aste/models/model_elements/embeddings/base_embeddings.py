from typing import Any, Union, Dict

from transformers import DebertaModel, AutoModel

from ....models import BaseModel


class BaseEmbedding(BaseModel):
    def __init__(self, embedding_dim: int, config: Dict, model_name: str = 'Base embedding model'):
        super(BaseEmbedding, self).__init__(model_name=model_name, config=config)
        self.model: Any = None
        self.embedding_dim: int = embedding_dim

    def get_transformer_encoder_from_config(self) -> Union[DebertaModel, AutoModel]:
        if 'deberta' in self.config['encoder']['transformer']['source']:
            return DebertaModel.from_pretrained(self.config['encoder']['transformer']['source'])
        elif 'bert' in self.config['encoder']['transformer']['source']:
            return AutoModel.from_pretrained(self.config['encoder']['transformer']['source'])
        else:
            raise Exception(
                f"We do not support this transformer model {self.config['encoder']['transformer']['source']}!")
