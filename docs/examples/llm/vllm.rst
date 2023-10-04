LLM Inference
=============


Server
------

`vLLM <https://github.com/vllm-project/vllm>`_ comes with its own server entry point that mimicks OpenAI's API.
It is very easy to setup and supports a wide range of models through Huggingfaces.


.. code-block:: 

   # sbatch inference_server.sh -m MODEL_NAME -p WEIGHT_PATH -e CONDA_ENV_NAME_TO_USE
   sbatch inference_server.sh -m Llama-2-7b-chat-hf -p /network/weights/llama.var/llama2/Llama-2-7b-chat-hf -e base


By default the script will launch the server on an rtx8000 for 15 minutes.
You can override the defaults by specifying arguments to sbatch.


.. code-block:: 

   sbatch --time=00:30:00 inference_server.sh -m Llama-2-7b-chat-hf -p /network/weights/llama.var/llama2/Llama-2-7b-chat-hf -e base

.. note::

    We are using job comment to store hostname, port and model names,
    which enable the client to automatically pick them up on its side.


.. literalinclude:: inference_server.sh
    :language: bash


Client
------

Becasue vLLM replicates OpenAI's API, the client side is quite straight forward.
Own OpenAI's client can be reused. 

.. warning::
   
   The server takes a while to setup you might to have to wait a few minutes
   before the server is ready for inference.

   You can check the job log of the server.
   Look for 


.. note::

   We use squeue to look for the inference server job to configure the 
   url endpoint automatically.

   Make sure your job name is unique!

.. literalinclude:: client.py
    :language: python
