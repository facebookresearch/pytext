# Getting Started on a Cloud VM

This document should help you quickly get up and running with a GPU attached Virtual Machine in the major cloud services. This guide will cover all the setup work you have to do in order to be able to easily install PyText.
Note that while these instructions worked when they were written, they may become incorrect or out of date. If they do, please send us a Pull Request!

## Amazon Web Services 
**Coming Soon**

## Google Cloud Engine

*If you have problems launching your VM, make sure you have a non-zero gpu quota, [click here to learn about quotas](https://cloud.google.com/compute/quotas#requesting_additional_quota)*

This guide uses [Google's Deep Learning VM](https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning) as a base. 

* Click "Launch on Compute Engine"
* Configure the VM:
  * Configure the hardware to match your desire/budget, the size you need will be determined by the data and model you use, but basic tasks like the ATIS tuturial can be run on the default 2 CPU 13.5 GB ram setup.
  * For Framework, select one of the Base images, rather than one with a framework pre-installed. Note which version of CUDA you choose for later.
  * When you're ready, click "Deploy"
  * When your VM is done loading, you can SSH into it from the GCE Console
* Install Python 3.6 (based on [this RoseHosting blog post](https://www.rosehosting.com/blog/how-to-install-python-3-6-4-on-debian-9/)):
  * `$ sudo nano /etc/apt/sources.list`
  * add `deb http://ftp.de.debian.org/debian testing main` to the list
  * `$ echo 'APT::Default-Release "stable";' | sudo tee -a /etc/apt/apt.conf.d/00local`
  * `$ sudo apt-get update`
  * `$ sudo apt-get -t testing install python3.6`
  * `$ sudo apt-get install python3.6-venv`
* Follow the [install instructions](INSTALL.md) for Ubuntu/Debian
* Follow the [Getting Started Instrutions](README.md) to download and setup PyText


## Microsoft Azure
**Coming Soon**