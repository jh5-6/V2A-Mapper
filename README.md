# V2A-Mapper

This repository contains the implementation of Regression MLPs for V2A-Mapper Strategies, which is part of the "V2A-Mapper: A Lightweight Solution for Vision-to-Audio Generation by Connecting Foundation Models" project. <b> Please note that this repository is unofficial. </b> [Paper](https://arxiv.org/abs/2308.09300), [Project page](https://v2a-mapper.github.io/)


## Method
![image](https://github.com/jh5-6/V2A-Mapper/assets/82092205/115080ce-d481-450a-9e99-eb67be0d3944)
Our lightweight method only requires the training of a V2A-Mapper to bridge the domain gap between the vision representative FM CLIP and the audio generative FM AudioLDM. The V2A-Mapper is supervised by the audio representative FM CLAP to learn the translation from visual space to auditory space. Leveraging the generalization and knowledge transfer ability of foundation models, the V2A-Mapper is trained with the same modestly sized dataset but the overall system can achieve much better performance.


## Results 
Visualize the domain gap between CLIP image space and CLAP audio space


<b>BEFORE Training</b>


<img src="https://github.com/jh5-6/V2A-Mapper/assets/82092205/060ef862-5bb0-413f-a963-216046235490" width="45%"> 

<b>AFTER Training</b>


<img src="https://github.com/jh5-6/V2A-Mapper/assets/82092205/3fec3a40-7515-484d-ab1e-969487d37461" width="45%">


## Web App
The web APP currently only support Text-to-Audio generation. For full functionality please refer to the Commandline Usage
1. Start the web application (powered by Gradio)
```python app.py```
2. A link will be printed out. Click the link to open the browser and play.

## Reference 
Part of the code is borrowed from the following repos. We would like to thank the authors of these repos for their contribution.
[AudioLDM](https://github.com/LAION-AI/CLAP](https://github.com/haoheliu/AudioLDM)
