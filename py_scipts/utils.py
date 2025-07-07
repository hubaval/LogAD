from enum import Enum
import numpy as np
import pandas as pd
import torch
import py_scipts.credentials as credentials


def get_peft_config():
    from peft import LoraConfig, TaskType
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        inference_mode=False,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
        )


def get_quant_config(quantization):
    if quantization == "bnb":
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
            )
    else:
        return None


def show(i, df):
    """
    Print line content to compare between target and predicted
    :param i:
    :return:
    """
    # Display setting to get all lines
    # pd.set_option('display.max_colwidth', None)
    # pd.reset_option('display.max_colwidth')

    print(f"""
TARGET
{repr(df.iloc[i]['full_target_text'])}

PREDICTED
{repr(df.iloc[i]['full_predicted'])}

SUMMARY
{df.iloc[i]}""")


def cut_timestamp(log, n_split):
    """
    Cut timestamps in log
    :param n_split:
    :param log:
    :return:
    """
    if n_split is None:
        return log, np.nan

    log_split = log.split(" ", maxsplit=n_split)
    log_message = log_split[-1]
    timestamp = ' '.join(log_split[0:n_split])
    return log_message, timestamp


def process_cut(logs, n_split):
    """
    Process log list for cut timestamp
    :param n_split:
    :param logs:
    :return:
    """
    df = pd.DataFrame(logs, columns=["log"])
    df[['log_message', 'timestamp']] = df['log'].apply(lambda x: pd.Series(cut_timestamp(x, n_split)))
    return df['log_message'].tolist()


def load_data(path="../data/mapping.json", correct_to_float=None):
    try:
        from datasets import load_from_disk, Dataset
        import os
        if os.path.isdir(path):
            ds = load_from_disk(path)
            try:
                dtf = Dataset.to_pandas(ds)
            except AttributeError:
                print('Cannot convert to dataframe, no split selected ? returning Datasets')
                return ds
        elif os.path.isfile(path):
            if "json" in path:
                dtf = pd.read_json(path)
            else:
                dtf = pd.read_csv(path, header=None, sep="\r")
                dtf.rename(columns={0: 'Text'}, inplace=True)
        else:
            raise ValueError(f"Path {path} does not exist.")

    except Exception as e:
        raise ValueError(f"Error processing file: {e}")

    if correct_to_float is not None:
        dtf[correct_to_float] = dtf[correct_to_float].astype(np.float32)

    return dtf


def check(total_metrics, total_samples):
    print(total_metrics / total_samples if total_samples > 0 else 0)


def get_key_from_value(value):
    return next((key for key, enum_value in EnumDatasource.__members__.items()
                 if enum_value.value == value), None)


def get_enum_timestamp_value(value):
    key = get_key_from_value(value)
    if key:
        return EnumTimestampCut[key].value
    return None

def batch_process(inputs, targets, chunk_size):
    """
    Function to process multiple inputs in chunks
    :param inputs:
    :param targets:
    :param chunk_size:
    :return:
    """
    for i in range(0, len(inputs), chunk_size):
        yield inputs[i:i + chunk_size], targets[i:i + chunk_size]


class EnumEnv(Enum):
    """
    Enum class for environment variables
    """
    TRAINING = "training"
    TESTING = "testing"
    DEVELOPMENT = "development"
    EVALUATION = "evaluation"


class EnumDatasource(Enum):
    """
    Datasource class for datasets
    """
    THUNDERBIRD = "THUNDERBIRD"
    BGL = "BGL"
    SPIRIT = "SPIRIT"
    NSL_KDD = "NSL-KDD"
    IDS2017 = "IDS 2017"
    IDS2018 = "IDS18"
    UPDATE = "UPDATE"


class EnumTimestampCut(Enum):
    """
    Number of tokens for timestamp for datasets
    """
    THUNDERBIRD = 6
    BGL = 5
    SPIRIT = 6
    NSL_KDD = None
    IDS2018 = 2
    IDS2017 = None


class CustomModel:
    """
    Custom import of models
    """

    def __init__(self, model_name, device=None, environment=EnumEnv.DEVELOPMENT, dtype="auto", quantize="bnb"):

        print(f"CustomModel start ---- {model_name}")

        self.device = device or (torch.cuda.current_device() if torch.cuda.is_available() else 'auto')
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.env = environment
        self.dtype = dtype
        self.quantize = quantize

        if self.env == EnumEnv.DEVELOPMENT:
            self.local = False
            from huggingface_hub import login
            login(token=credentials.hf_bis)
        elif self.env == EnumEnv.TRAINING:
            self.local = True
        elif self.env == EnumEnv.TESTING:
            self.local = True
        elif self.env == EnumEnv.EVALUATION:
            self.local = True
        else:
            raise ValueError("CustomModel error ---- Invalid environment")

        print(f"CustomModel setup ---- Working on device : {self.device}")
        print(f"CustomModel setup ---- Selected environment : {self.env}")

    def load(self):
        # Load the model and tokenizer from a saved state
        self.load_model()
        self.load_tokenizer()
        return self.model, self.tokenizer

    def load_model(self, train=False):
        # Load the model
        from transformers import AutoModelForCausalLM
        print("CustomModel model ---- Loading model")

        quantization_config = get_quant_config(self.quantize)
        if quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=quantization_config, device_map=self.device, local_files_only=self.local, torch_dtype=self.dtype)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.device, local_files_only=self.local, torch_dtype=self.dtype)
        print("CustomModel model ---- Done")
        return self.model.train() if train else self.model

    def load_tokenizer(self):
        # Load the tokenizer
        from transformers import AutoTokenizer
        print("CustomModel tokenizer ---- Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left', local_files_only=self.local)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("CustomModel tokenizer ---- Done")
        return self.tokenizer

    def predict(self, inputs, max_new_tokens=128, stop_strings=['\n', '###'], temperature=0.2):
        # Verify if model and tokenizer are loaded
        assert self.model is not None, "CustomModel error ---- Model must be loaded before prediction"
        assert self.tokenizer is not None, "CustomModel error ---- Tokenizer must be loaded before prediction"

        # Predict next log from given input (context)
        if isinstance(inputs, (pd.DataFrame, pd.Series, np.ndarray)):
            inputs = inputs.tolist()

        prompts = []
        if isinstance(inputs, list):
            for input in inputs:
                prompts.append(f"### Context: {input}\n### Next: ")
        else:
            prompts = f"### Context: {inputs}\n### Next: "

        inputs_tokenized = self.tokenizer(prompts, padding=True, return_tensors='pt').to('cuda' if self.device == "auto" else self.device)

        with (torch.no_grad()):
            inputs_tokenized['input_ids'] = inputs_tokenized['input_ids']
            outputs = self.model.generate(**inputs_tokenized, max_new_tokens=max_new_tokens, stop_strings=stop_strings, tokenizer=self.tokenizer, temperature=temperature,
                                          pad_token_id=self.tokenizer.eos_token_id)

        tokens = outputs[:, inputs_tokenized['input_ids'].shape[1]:]
        predicted = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

        return predicted

    def get_embeddings(self, texts):
        # Verify if model and tokenizer are loaded
        assert self.model is not None, "CustomModel error ---- Model must be loaded before prediction"
        assert self.tokenizer is not None, "CustomModel error ---- Tokenizer must be loaded before prediction"

        inputs = self.tokenizer(texts, padding=True, return_tensors='pt').to('cuda' if self.device == "auto" else self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, pad_token_id=self.tokenizer.eos_token_id)['hidden_states'][-1].mean(dim=1).squeeze().cpu().numpy()
        return outputs

    def cosine_similarity_numpy(self, a, b):
        a_emb = np.array(self.get_embeddings(a))
        b_emb = np.array(self.get_embeddings(b))
        return np.dot(a_emb, b_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(b_emb))
