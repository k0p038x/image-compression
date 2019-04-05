Implementation of ComCNN and FSRCNN


Model:

1. ComCNN
2. RecCNN (FSRCNN)

FSRCNN:

1. First Conv layer - d filters, kernel size (5,5)
2. Second Conv layer - s filters, kernel size (1,1)
3. 'm' Conv layers - s filters, kernel size (3,3)
4. Conv layer - d filters, kernel size (1,1)
5. ConvTranspose layer - kernel size (9,9) , stride size (4,4)

FSRCNN parameters: 

d = 56
s = 12
m = 4

Dataset : 
1. Dataset consists of 5000 images of size 256x256
2. train set - 4000 images
3. validation set - remaining 1000 images

Training: 
1. We trained RecCNN and ComCNN initially (instead of assigning random weights). Each model (RecCNN followed by ComCNN) trained over 25 epochs
2. Implemented the algo presented in the paper.
Parameters: iterations = 15, epochs_per_each_iteration = 10

Results:
1. Test dataset contains 13 images from Set14

Average    : 0.8696310078751335
