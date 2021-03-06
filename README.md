# srgan_cosmo2d 
Application of SRGAN to enhance simulations of 2D distribution of dark matter in the Universe.
This code takes `LR` image of size 128^2 and generates `SR` image of size 512^2. `HR` is the ground truth high-resolution image, `ILR` is the interpolated LR image to 512^2 resolution.
The example below was obtained after 15-hour training on one A100.
<br>
<br> Fig. 1. Top: LR, ILR. Bottom: SR, HR. Magenta line defines `skewer` which passes through the brightest pixel on the HR image.
![plot1](docs/img5_f4.png )


<br> Fig. 2. Pixel intensity along the skewer.
![plot2](docs/img5_f5.png ) 

<br> Fig. 3. Denisty histogram and power spectrum. Top: absolute, bottom: relative.
![plot2](docs/img5_f6.png ) 
The code used here was derived from https://github.com/Lornatang/SRGAN-PyTorch

 
