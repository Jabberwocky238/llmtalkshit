import copy 
from typing import Optional
import torch
import torch.nn as nn

from .PretrainedConfig import PretrainedConfig
from .LlamaConfig import LlamaConfig
_init_weights = True
from transformers.generation.utils import GenerationConfig
from transformers.generation.utils import GenerationMixin

# class LlamaPreTrainedModel(PreTrainedModel):
class LlamaPreTrainedModel(nn.Module, GenerationMixin):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "causal_mask"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    main_input_name = "input_ids"
    device = torch.device("cuda")

    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation.
        # Alternativelly, the model can also have a custom `generate` function.
        if "GenerationMixin" in str(cls.prepare_inputs_for_generation) and "GenerationMixin" in str(cls.generate):
            return False
        return True

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__()
        print("LlamaPreTrainedModel __init__")
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # Save config and origin of the pretrained weights if given in model
        # config = self._autoset_attn_implementation(
        #     config, torch_dtype=torch.get_default_dtype(), check_device_map=False
        # )
        self.config = config

        self.name_or_path = config.name_or_path
        self.warnings_issued = {}
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        # Overwrite the class attribute to make it an instance attribute, so models like
        # `InstructBlipForConditionalGeneration` can dynamically update it without modifying the class attribute
        # when a different component (e.g. language_model) is used.
        # self._keep_in_fp32_modules = copy.copy(self.__class__._keep_in_fp32_modules)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):
        # if self.config._attn_implementation == "flash_attention_2" and cache_cls == StaticCache:
        #     raise ValueError(
        #         "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
        #         "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        #     )

        if max_cache_len > self.model.causal_mask.shape[-1] or self.device != self.model.causal_mask.device:
            causal_mask = torch.full(
                (max_cache_len, max_cache_len), fill_value=True, device=self.device, dtype=torch.bool
            )
            self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)

        for layer in self.model.layers:
            weights = layer.self_attn.o_proj.weight
            layer.self_attn.past_key_value = cache_cls(
                self.config, max_batch_size, max_cache_len, device=weights.device, dtype=weights.dtype
            )

    def _reset_cache(self):
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None


    def post_init(self):
        print("LlamaPreTrainedModel post_init")
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()
        # self._backward_compatibility_gradient_checkpointing()

    def init_weights(self):
        print("LlamaPreTrainedModel init_weights")
        """
        If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # Prune heads if needed
        # if self.config.pruned_heads:
        #     self.prune_heads(self.config.pruned_heads)

        if _init_weights:
            # Initialize weights
            self.apply(self._initialize_weights)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways
            self.tie_weights()

    def tie_weights(self):
        print("LlamaPreTrainedModel tie_weights")
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
        weights instead.
        """
        # if getattr(self.config, "tie_word_embeddings", True):
        #     output_embeddings = self.get_output_embeddings()
        #     if output_embeddings is not None:
        #         self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

        # if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
        #     if hasattr(self, self.base_model_prefix):
        #         self = getattr(self, self.base_model_prefix)
        #     self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    def _initialize_weights(self, module: nn.Module):
        print("LlamaPreTrainedModel _initialize_weights")
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(module, "_is_hf_initialized", False):
            return
        self._init_weights(module)
        module._is_hf_initialized = True

    def _init_weights(self, module):
        print("LlamaPreTrainedModel _initialize_weights")
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
