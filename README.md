# Faster Whisper - Whispering LLama

Setup
Clone the repo

```bash
git clone https://github.com/Srijith-rkr/Whispering-LLaMA
cd WHISPERing-LLaMA
```

Install dependencies with Anaconda

```bash
conda env create -f environment.yml
```
Or you can also use the requirements.txt as
```bash
pip install -r requirements.txt
```


- To obtain the pre-trained Alpaca weights, please refer [here](https://github.com/tatsu-lab/stanford_alpaca#recovering-alpaca-weights). You can then use convert_hf_checkpoint.py on Whispering-LLaMA repository to rename the state_dict the [lit-llama](https://github.com/Lightning-AI/lit-llama) implementation
- Or you can use the Alpaca weights hosted in HuggingFace [Huggin Face/Whispering-LLaMA](https://huggingface.co/Srijith-rkr/Whispering-LLaMA). Refer to demo.py on Whispering-LLaMA repository how to use them.
- Obtain pretrained tokenizer model from [hugging face](https://huggingface.co/Srijith-rkr/Whispering-LLaMA/tree/main)
- Obtain pretrained lit-llama model from [hugging face](https://huggingface.co/Gary3410/pretrain_lit_llama/blob/main)



# Dataset preparation
#### `data_preparation/`

To Generate json files for respective podcast run as
```bash
python3 generate_json_from_csv.py 
```
After following the three steps stated below:

- Create a csv directory to store corresponding csv files of each podcast. An example can be found in data_preparation directory
- Create audio data directory to store the audio files. 
- Create audio features directory to save the generated files (.json files)

To generate audio features for respective podcast load the json file saved in previous Step and provide filepath to save the generated tensor file (.pt files) and run the file as
```bash
python3 generate_audio_features.py
```

# Training and inference
Upon completion of dataset preparation, move the adapter_copy.py to the training directory within Whispering-LLama directory

- Provide lit-llama petrained path and run as
  ```bash
  python3 adapter_copy.py --lr 1e-3 -d 1 --data ac
  ```
  You can configure the following flags.
    ```
    --lr: learning rate (1e-3 is recommended)
    --d: Number of GPUs you are using to run the DDP strategy (You can uncomment lines in the code to switch to DeepSpeed)
    --data: Path to your dataset, example ac_train.pt, ac_test.pt
    ```
- In adapter_copy.py tune the parameters: batch size, micro_batch_size, max_seq_length, max_input_length based on the available resource
- Save the adapter checkpoint

Finally, run the whispering_LLama inference. 

```bash
python3 llama_whisper_adapter_inference.py \
    --pretrained_path 'model/alpaca_a.pth model/alpaca_b.pth model/alpaca_c.pth' \
    --tokenizer_path 'model/tokenizer.model' \
    --data 'audio_features/ac048_2007-08-06_train.pt' \
    --save_dir 'inference_result' \
    --root 'model/adapter_checkpoints'
 ```  
