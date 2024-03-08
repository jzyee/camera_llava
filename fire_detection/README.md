Links to youtube videos containing fires:
- police webcam: https://www.youtube.com/watch?v=Dapl4VXKWnk
- webcam inside house showing the house slowly catchingon fire, mostly smoke(video quality is not so good): https://www.youtube.com/watch?v=JTlGFsP4JeQ
- warehouse catching on fire: https://www.youtube.com/watch?v=ctN3hD3C5Kc



1)look into a search function for each frame:
- the prompt that the user puts in, checks every single frame
- then the user can use a checkbox to select which values the user wants to see, the values correspond to a frame
- can give visualizations to the cluters from the selected results if possible

2)find videos of traffic:
- use case:
 * e.g check which frame has a person with a helmet

3)find videos of sleepy driver:
 - use case:
    * e.g monitoring the live stream if the drivers are sleeping
    * check for precursors to sleep
        * like the bus driver eyes are closed or almost closed

4) need to benchtest somehow on some online  labelled dataset

important
1) alert detection using pre-defined prompts
2) investigation func (chatbot and search func)



# how to get the code running


Need to download the following files
- ggml-model-q5_k.gguf
- mmproj-model-f16.gguf

from: https://huggingface.co/mys/ggml_llava-v1.5-7b/tree/main



command line code to get the llama cpp server running for the gui
```
python -m llama_cpp.server --model ggml-model-q5_k.gguf --clip_model_path mmproj-model-f16.gguf --chat_format llava-1-5 --n_gpu_layers 1 --n_threads 8
```
*note: it is using the conda env: llava-server to run this (needed pyhton 3.9)

steps :
1) activate the codna env llava-server
2) cd into fire_detection
3) run the above command line code
4) run the gui code

# how to change the parameters of the mdoel in the code

for the llama-cpp-python models: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/

the llmaa-cpp-python models are:
1) the QA model
2) the surveilance model

the pytorch models are:
1) the searching model