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

coastguard : 0.8603603806975104
monarch    : 0.8894539961733375
foreman    : 0.9558246475128839
man        : 0.8898917771659295
pepper     : 0.8795445313324509
zebra      : 0.8474724554933961
bridge     : 0.8415195638030845
comic      : 0.8422174709009371
barbara    : 0.8032147384335037
lenna      : 0.9104160738774046
face       : 0.9257891156939335
ppt3       : 0.8399573551388464
flowers    : 0.8195409961535151
--------------------------------
Average    : 0.8696310078751335
