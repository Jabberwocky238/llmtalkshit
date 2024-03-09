import copy
import inspect 
from typing import Callable, List, Optional, Union
import warnings
import torch
import torch.nn as nn

from LlamaForCausalLM.StaticCache import StaticCache

from .PretrainedConfig import PretrainedConfig
from .LlamaConfig import LlamaConfig
_init_weights = True
from transformers.generation.utils import GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.generation.stopping_criteria import MaxLengthCriteria
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
from transformers.generation.utils import GenerateOutput, GenerateNonBeamOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers import AutoTokenizer

# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model = LlamaForCausalLM.from_pretrained(model_name, cache_dir='../')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir='E:/1-code/Python/')
# models--TinyLlama--TinyLlama-1.1B-Chat-v1.0

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
        return True
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        print("LlamaPreTrainedModel generate")
        synced_gpus = False
        print(generation_config)
        print(self.generation_config)
        generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        self._validate_model_kwargs(model_kwargs.copy())
        # print(model_kwargs) # {}

        # 2. Set generation parameters if not already defined
        # logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        # stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        # print("logits_processor", logits_processor)
        # print("stopping_criteria", stopping_criteria)
        # logits_processor = []
        # stopping_criteria = []

        # print("pad_token_id", generation_config.pad_token_id)None
        # print("eos_token_id", generation_config.eos_token_id)2
        # print("bos_token_id", generation_config.bos_token_id)1
        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            print(f"Setting `pad_token_id` to `eos_token_id`:{generation_config.eos_token_id} for open-end generation.")
            generation_config.pad_token_id = generation_config.eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        # print("inputs before _prepare_model_inputs", inputs)
        # inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        #     inputs, generation_config.bos_token_id, model_kwargs
        # )
        # print("inputs_tensor after _prepare_model_inputs", inputs_tensor)
        # print(model_input_name)input_ids

        # inputs_tensor没有变化
        inputs_tensor = inputs
        # print(inputs_tensor)
        
        # print(inputs_tensor.size())
        # print(inputs_tensor.size()[-1])
        # batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        model_kwargs["use_cache"] = generation_config.use_cache

        # accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        # requires_attention_mask = "encoder_outputs" not in model_kwargs
        # print("accepts_attention_mask", accepts_attention_mask)
        # print("requires_attention_mask", requires_attention_mask)
        # if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        #     print("if model_kwargs.get() is None and requires_attention_mask and accepts_attention_mask:")
        #     model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
        #         inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        #     )

        # decoder-only models should use left-padding for generation
        model_input_name = "input_ids"
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
        input_ids_length = input_ids.size()[-1]
        # if streamer is not None:
        #     streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        # input_ids_length = input_ids.size()[-1]
        # has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        # generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        # if we don't pass `past_key_values` and a cache_implementation is specified
        # NEED_SETUP_CACHE_CLASSES_MAPPING = {
        #     "static": StaticCache,
        # }
        # if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING and not model_kwargs.get(
        #     "past_key_values", False
        # ):
        #     cache_cls = NEED_SETUP_CACHE_CLASSES_MAPPING[generation_config.cache_implementation]
        #     if not callable(getattr(self, "_setup_cache", None)):
        #         raise ValueError(
        #             "The `generation_config` defines a `cache_implementation` that is not compatible with this model."
        #             " Make sure it has a `_setup_cache` function."
        #         )
        #     self._setup_cache(cache_cls, max_batch_size=batch_size, max_cache_len=generation_config.max_length)

        # self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        # generation_mode = self._get_generation_mode(generation_config, assistant_model)


        if self.device.type != input_ids.device.type:
            input_ids.to(self.device)

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = LogitsProcessorList()
        # prepared_logits_processor = self._get_logits_processor(
        #     generation_config=generation_config,
        #     input_ids_seq_length=input_ids_length,
        #     encoder_input_ids=inputs_tensor,
        #     prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        #     logits_processor=logits_processor,
        #     model_kwargs=model_kwargs,
        #     negative_prompt_ids=negative_prompt_ids,
        #     negative_prompt_attention_mask=negative_prompt_attention_mask,
        # )
        prepared_logits_processor.append(
            RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty)
        )
        print("prepared_logits_processor", prepared_logits_processor)

        # 9. prepare stopping criteria
        # prepared_stopping_criteria = self._get_stopping_criteria(
        #     generation_config=generation_config, stopping_criteria=stopping_criteria
        # )
        prepared_stopping_criteria = StoppingCriteriaList()
        prepared_stopping_criteria.append(
            MaxLengthCriteria(
                max_length=generation_config.max_length,
                max_position_embeddings=2048,
            )
        )
        print("prepared_stopping_criteria", prepared_stopping_criteria)
        # 10. go into different generation modes

        # 11. prepare logits warper
        logits_warper = self._get_logits_warper(generation_config)
        # logits_warper = LogitsProcessorList()
        # logits_warper.append(TemperatureLogitsWarper(generation_config.temperature))
        # logits_warper.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        # logits_warper.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        print("logits_warper", logits_warper)

        # 12. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 13. run sample
        return self.sample(
            input_ids,
            logits_processor=prepared_logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )
    
    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        
        # logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        # stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        # logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()

        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        # output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        # output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        # output_attentions = (
        #     output_attentions if output_attentions is not None else self.generation_config.output_attentions
        # )
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        # )
        # return_dict_in_generate = False

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        # this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:
            # print("sample while")
            # prepare model inputs
            # print(input_ids.size())
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # print("model_inputs", model_inputs['input_ids'])tensor([[8160]])
            temp: torch.Tensor = model_inputs['input_ids'][0]
            tmp = tokenizer.decode(temp, skip_special_tokens=False)
            print(tmp, end="")
            # print("model_inputs", model_inputs['position_ids'])tensor([[688]])
            # print("model_inputs", model_inputs['cache_position'])tensor([688])
            # print("model_inputs", model_inputs['use_cache'])True
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            # print("outputs", outputs['logits'].size())torch.Size([1, 1, 32000])

            # if synced_gpus and this_peer_finished:
            #     continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            # if return_dict_in_generate:

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            # input_ids会逐渐变长
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            # if streamer is not None:
            #     streamer.put(next_tokens.cpu())
            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            # )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, None):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        # if streamer is not None:
        #     streamer.end()

        return input_ids


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
