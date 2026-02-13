##### Containers from Nvidia cloud

One advantage of pulling containers from nvidia cloud is that nvidia software such as Apex or Dali are already installed on them.
In order to get those containers, you need to register on `ngc`_. After you log in, go on configuration and Generate an API Key.

.. image:: ngc-0.png

Once you get your API key, you need to add them into your bashrc:

     echo "export SINGULARITY_DOCKER_USERNAME='$oauthtoken'" >> ~/.bashrc
     echo "export SINGULARITY_DOCKER_PASSWORD=YOUR_API_KEY_HERE" >> ~/.bashrc

Then you can pull the container you want from the `nvidia page`_. For example to get pytorch:

     singularity pull docker://nvcr.io/nvidia/pytorch:19.02-py3

or if you want tensorflow:

     singularity pull docker://nvcr.io/nvidia/tensorflow:19.02-py3

> **WARNING**