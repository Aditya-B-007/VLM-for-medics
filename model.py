import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipImageProcessor, SiglipVisionModel
from transformers import BitsAndBytesConfig
import os
import json
from PIL import Image
from .config import Config
class VLMMetaArchitecture:
    pass

class VLM(VLMMetaArchitecture, nn.Module):
    def __init__(self, vision, llm, tokenizer):
        super().__init__()
        self.vision = vision
        self.llm = llm
        self.tokenizer = tokenizer
        self.configure_tokenizer()

        # define the connector layer
        self.connector = nn.Sequential(
            nn.Linear(vision.config.hidden_size, llm.config.hidden_size),
            nn.SiLU(),
            nn.Linear(llm.config.hidden_size, llm.config.hidden_size)
        )

        # weight tying uses the embedding layer as the lm_head
        self.llm.tie_weights()


    def freeze(self):
        # freeze all model parameters
        for param in self.parameters():
            param.requires_grad = False

        # unfreeze connector
        for p in self.connector.parameters():
          p.requires_grad = True

        # unfreeze llm embedding weights
        for p in self.llm.embed_tokens.parameters():
          p.requires_grad = True


    def configure_tokenizer(self):
        self.tokenizer.padding_side = 'right'
        self.tokenizer.add_tokens(["<img>", "</img>"])

        # resize token embedding matrix and lm_head
        self.llm.resize_token_embeddings(len(self.tokenizer))

    @property
    def device(self):
        return self.llm.device
    @property
    def img_token(self):
        return self.tokenizer.encode("<img>", add_special_tokens=False)[0]
    @property
    def end_img_token(self):
        return self.tokenizer.encode("</img>", add_special_tokens=False)[0]

    @property
    def embed_tokens(self):
        return self.llm.model.embed_tokens


    def prepare_inputs(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        images=None
    ):
        """
        Prepare the inputs for the model.
        """
        if images is None:
            return input_ids, attention_mask, inputs_embeds, labels #no-op; standard llm preprocessing

        bsz = len(images)

        vision_embeds = self.vision(images) # NOTE: WE STILL NEED TO DEFINE THIS !
        vision_embeds = self.connector(vision_embeds)

        # bos
        bos_token = torch.tensor(self.tokenizer.bos_token_id, device=self.device).unsqueeze(0)
        bos_embeds = self.embed_tokens(bos_token).repeat((bsz,1,1))

        # embed <img> and </img>
        img_token = torch.tensor(self.img_token, device=self.device).unsqueeze(0)
        img_embeds = self.embed_tokens(img_token).repeat((bsz,1,1))
        end_img_token = torch.tensor(self.end_img_token, device=self.device).unsqueeze(0)
        end_img_embeds = self.embed_tokens(end_img_token).repeat((bsz,1,1))

        if input_ids is not None:
            # embeddings
            text_embeds = self.embed_tokens(input_ids)
            input_ids = None

            inputs_embeds = torch.cat((bos_embeds, img_embeds, vision_embeds, end_img_embeds, text_embeds[:,1:,:]), dim=1)


            # attention_mask
            _, vis_len, _ = vision_embeds.shape
            additional_len = 1 + 1 #added <img> and </img>
            attention_mask = torch.cat((torch.ones((bsz, vis_len+additional_len), device=self.device), attention_mask), dim=1)


            # labels
            if labels is not None:
                labels = labels[:,1:]
                additional_len+=1 #added </s>
                labels_prefix = torch.tensor([-100]*(vis_len+additional_len), device = self.device) #tokens with -100 get ignored in loss fn

                labels_prefix = labels_prefix.repeat((bsz, 1))
                labels = torch.cat((labels_prefix, labels), dim=1)


        else:
            inputs_embeds = torch.cat((bos_embeds, img_embeds, vision_embeds, end_img_embeds), dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:-1], device=self.device)

        return None, attention_mask, inputs_embeds, labels


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        return_dict=True,
    ):
        """
        Pass the concatenated [img, text] embedding through the llm.
        """
        assert input_ids is not None or images is not None, "You can't forward without text and/or images!"

        input_ids, attention_mask, inputs_embeds, labels = self.prepare_inputs(input_ids, attention_mask, inputs_embeds, labels, images)

        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        images=None,
        **kwargs,
    ):
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        attention_mask = kwargs.pop("attention_mask", None)

        if images is not None:
              _, attention_mask, inputs_embeds, _ = self.prepare_inputs(input_ids, attention_mask, None, None, images)
        else:
            # if no images passed we just use the text embeddings
            inputs_embeds = self.llm.model.embed_tokens(input_ids)

        return self.llm.generate(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
class Vision(nn.Module):
    def __init__(self, vision, processor, r:int = 1):
        super().__init__()
        self.vision = vision
        self.image_processor = processor
        self.r = r #how many adjacent tokens to concatenate

        # reshape sequence
        setattr(self.vision.config, 'hidden_size', self.r*self.vision.config.hidden_size)

    @property
    def config(self):
        return self.vision.config
    @property
    def device(self):
        return self.vision.device

    def vit_forward(self, x):
        x = self.image_processor(x, return_tensors = 'pt')['pixel_values']
        x = self.vision(x.to(self.device), output_hidden_states=True)
        x = x['hidden_states'][-1] # which depth to choose hidden features from
        return x

    def forward(self, x, attention_mask = None):
        # vision-backbone
        x = self.vit_forward(x)
        b,s,_ = x.shape

        # concatenate adjacent tokens a la minigpt4-v2
        return x.reshape((b, s//self.r, -1))

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# language
class VLMModel:
    def __init__(self, weights_path: str = None):
        super().__init__()
        llm_id = Config.LLM_ID
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            #quantization_config=quant_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id)

        vision_id = Config.VISION_ID
        vision_backbone = SiglipVisionModel.from_pretrained(vision_id)
        self.img_processor=SiglipImageProcessor.from_pretrained(vision_id)

        # The Vision class wraps the backbone model
        vision_tower = Vision(vision_backbone, self.img_processor, r=Config.VISION_TOWER_R)

        # Create the VLM model architecture
        self.model = VLM(vision_tower, self.llm, self.tokenizer)

        # If a path to pre-trained (fine-tuned) weights is provided, load them
        if weights_path and os.path.exists(weights_path):
            print(f"Loading fine-tuned weights from {weights_path}...")
            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            print("Weights loaded successfully.")

        # Move the entire model to the GPU
        self.model.to(Config.DEVICE)