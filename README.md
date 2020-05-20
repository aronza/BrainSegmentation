#       3D U-Net for Volumetric Segmentation of Brain MRIs
###                      By Arda Turkmen

#       Instructions

##      Installation
    Python and it's dependencies are required.
    
    After downloading the repository one could run this command to install dependencies:
        pip install -r BrainSegmentation/requirements.txt 
    
##      Executing
    To predict an image use the predict.py script in src folder. For this script there are several arguments to control the prediction. 
    One important thing to note is that either a type needs to be supplied to decide what to segment or a custom model's path has to be supplied with the model argument. 
    Patch-size controls the height of 2D stacks to be fed into the model.
    
    Images are expected to be in .nii.gz format.

    optional arguments:
      -h, --help            show this help message and exit
      --type TYPE, -t TYPE  mask = Brainmask segementation, wm = WhiteMatter
                            segmentation (default: None)
      --model FILE, -m FILE
                            Specify the file in which the model is stored
                            (default: False)
      --input INPUT, -i INPUT
                            directory of input images (default: None)
      --output OUTPUT, -o OUTPUT
                            directory of output files (default: None)
      -p P, --patch-size P  Patch Size (default: 16)
      -n NAME, --name NAME  Postfix to append to output filenames (default: OUT)
      -v, --verbose         Log more detail (default: False)

    train.py can be used to train a custom model. Use --help for further details on this scripts arguments. If using a custom dataset make sure to supply a torch.utils.data.DataSet class.

    eval.py defines evaluation for a model to be used for both validation and testing

    Rest of the scripts including the model definition are in unet3d folder.

    Model is based on
    Çiçek, Ö., Abdulkadir, A., Lienkamp, S.S., Brox, T., & Ronneberger, O. (2016).
    3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. ArXiv, abs/1606.06650.
