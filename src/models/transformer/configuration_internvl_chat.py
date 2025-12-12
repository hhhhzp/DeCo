# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy
from types import SimpleNamespace
from typing import Optional, Tuple

from transformers import AutoConfig, LlamaConfig, Qwen2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig
from transformers.utils import logging

from .configuration_intern_vit import InternVisionConfig

logger = logging.get_logger(__name__)


class WanTransformer3DConfig(PretrainedConfig):
    """
    WanTransformer3DModel 的配置类。
    """

    # model_type 是一个好习惯，便于标识模型类型
    model_type = "wan_transformer_3d"

    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # 将所有参数保存为类的属性
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.cross_attn_norm = cross_attn_norm
        self.qk_norm = qk_norm
        self.eps = eps
        self.image_dim = image_dim
        self.added_kv_proj_dim = added_kv_proj_dim
        self.rope_max_seq_len = rope_max_seq_len


class InternVLChatConfig(PretrainedConfig):
    model_type = 'internvl_chat'
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        pad2square=False,
        select_layer=-1,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        dynamic_image_size=False,
        use_thumbnail=False,
        ps_version='v1',
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        vae_downsample_ratio=None,
        **kwargs,
    ):
        vision_config = kwargs.pop('vision_config', vision_config)
        llm_config = kwargs.pop('llm_config', llm_config)
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {'architectures': ['InternVisionModel']}
            logger.info(
                'vision_config is None. Initializing the InternVisionConfig with default values.'
            )

        if llm_config is None:
            # TODO: There might still be a bug in transformers version 4.44 and above.
            llm_config = {'architectures': ['']}
            logger.info(
                'llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).'
            )

        self.vision_config = InternVisionConfig(**vision_config)
        if llm_config['architectures'][0] == 'LlamaForCausalLM':
            self.llm_config = LlamaConfig(**llm_config)
        elif llm_config['architectures'][0] == 'Qwen2ForCausalLM':
            self.llm_config = Qwen2Config(**llm_config)
        elif llm_config['architectures'][0] == 'Qwen2ForUnifiedCausalLM':
            self.llm_config = Qwen2Config(**llm_config)
        elif llm_config['architectures'][0] == 'UnifiedForCausalLM':
            self.llm_config = Qwen2Config(**llm_config)
            # print(self.llm_config)
            # print(kwargs)
            if (
                hasattr(self.llm_config, "dit_config")
                and self.llm_config.dit_config is not None
            ):
                self.llm_config.dit_config = WanTransformer3DConfig(
                    **self.llm_config.dit_config
                )
            else:
                self.llm_config.dit_config = WanTransformer3DConfig(
                    **kwargs["dit_config"]
                )
        else:
            self.llm_config = CONFIG_MAPPING["qwen2"]()
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.vae_downsample_ratio = vae_downsample_ratio
        self.hidden_size = self.llm_config.hidden_size
        # By default, we use tie_word_embeddings=False for models of all sizes.
        self.tie_word_embeddings = False
        self.llm_config.tie_word_embeddings = self.tie_word_embeddings

        logger.info(f'vision_select_layer: {self.select_layer}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
        logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail
        output['ps_version'] = self.ps_version
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch

        return output
