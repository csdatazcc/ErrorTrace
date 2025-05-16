# ErrorTrace

This is the official code repository for **ErrorTrace**.

## Environment

All our programs were developed and tested under **Python 3.9.0**.  
Please ensure you are using the correct Python version before running the code.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Ollama Setup

Our base models are all downloaded via [Ollama](https://ollama.com/).  
To use our code, you need to install and set up Ollama on your local machine.

You can follow the official installation guide here:  
👉 [https://ollama.com/](https://ollama.com/)

Once installed, make sure to pull the required models before running the program.  
For example:

```bash
ollama run llama3
```

## Dataset

We use publicly available benchmark datasets in this project.  
All datasets are placed in the `data/` folder.

Please note:  
You will need to **generate erroneous data yourself** based on the dataset you choose to use with our method.  
Our code does not include automatic error generation, as this process may vary depending on the dataset and task.


