# speech_error_detection
# Rare event detection


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


- To obtain the pre-trained Alpaca weights, please refer [here](https://github.com/tatsu-lab/stanford_alpaca#recovering-alpaca-weights). You can then use convert_hf_checkpoint.py to rename the state_dict the [lit-llama](https://github.com/Lightning-AI/lit-llama) implementation
- Or you can use the Alpaca weights hosted in HuggingFace [Huggin Face/Whispering-LLaMA](https://huggingface.co/Srijith-rkr/Whispering-LLaMA). Refer to demo.py on how to use them.
-Obtain pretrained tokenizer model from [hugging face](https://huggingface.co/Srijith-rkr/Whispering-LLaMA/tree/main)
-Obtain pretrained lit-llama model from [hugging face](https://huggingface.co/Gary3410/pretrain_lit_llama/blob/main)



# Dataset preparation
#### `data_preparation/`

To Generate json files for respective podcast run as
```bash
python3 generate_json_from_csv.py 
```
After following the three steps stated below:
    -Create a csv directory to store corresponding csv files of each podcast. An example can be found in data_preparation directory
    -Create audio data directory to store the audio files. 
    -Create generated audio features directory to save the generated files (.json files)

To generate audio features for respective podcast run as
```bash
python3 generate_audio_features.py
```
Prior to ruuning the code load the json file saved in Step 1 and provide filepath to save the generated tensor file (.pt files)

# Training and inference
Upon completion of step 1 and step 2, move the adapter_copy.py to the training directory within Whispering-LLama directory
    - Provide lit-llama petrained path and run the adapter_copy.py --lr 0.001 -d 1 --data ac (specify learning rate, number of gpu and data)
    - Change batch size, model size, token length based on the resource
    - Need train and test files, example ac_train.pt, ac_test.pt
    - Save the adapter checkpoint
    
2) Finally, run the whispering_LLama inference. 

```bash
python3 llama_whisper_adapter_inference.py \
    --pretrained_path 'model/alpaca_a.pth model/alpaca_b.pth model/alpaca_c.pth' \
    --tokenizer_path 'model/tokenizer.model' \
    --data '/home/user/Documents/NEU_SFU/speech_error_classification/Whispering-LLaMA/audio_features/gs_inferences/ac048_2007-08-06_train.pt' \
    --save_dir '/home/user/Documents/NEU_SFU/speech_error_classification/Whispering-LLaMA/inference_result' \
    --root '/home/user/Documents/NEU_SFU/speech_error_classification/Whispering-LLaMA/model/adapter_checkpoints'
 ```  
