### WAIT Datasets
Download the WAIT animation and illustration datasets using the following command. 

This will download the dataset used in the experiments to current directory.
Check for other options in the "openlibraryImageDownloaderMain.py" 

```bash
python openlibraryImageDownloaderMain.py --openlib_username <username> --openlib_password <pass>
```

To train a model on illustration datasets for style transfer, 
you need to create a data folder with two subdirectories `trainA` and `trainB` that contain images from domain A (e.g. natural images) and B (e.g. Korky Paul). 
You can test your model on your training set by setting `--phase train` in `test.py`. 
You can also create subdirectories `testA` and `testB` if you have test data.
We use only `testA` for style transfer tests.
We use natural images from CycleGan art datasets such as `monet2photo` or `vangogh2photo`. 
Photo images are same in these datasets so you could pick one and download it.   

