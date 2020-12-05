# hand-gesture-recognition
Translating hand gestures to English letters

## Roles
- Bach: generate data + preprocess data + model
- Dat: generate data + web app 
- H.Nam: generate data + web app
- P.Nam: test models

## Dataset:
  - Train data: 
      - sign_mnist_train: https://www.kaggle.com/datamunge/sign-language-mnist
        - ![letters stat 1](./pics/sign_mnist_original_stat1.png)
        - ![letters stat 2](./pics/sign_mnist_original_stat2.png)
        - ![letters preview](./pics/sign_mnist_original_preview.png)
      - custom: https://drive.google.com/file/d/1G9HsXqpyc7MHf9C9lvWrM1LK20iHj94K/view
        - indoor capture (4 lightning schemes) + on-class capture (1 lightning schemes)
        - ![letters stat 1](./pics/custom_stat1.png)
        - ![letters stat 2](./pics/custom_stat2.png)
        - ![letters preview](./pics/custom_preview.png) 

## Model: 
  - 25GB RAM - GPU: https://colab.research.google.com/drive/1NlmPOHSN26Brrl51WAZfgiof8m7lJJMb?usp=sharing
  - ref: https://www.kaggle.com/sayakdasgupta/sign-language-classification-cnn-99-40-accuracy

## Evaluation
  - Test dataset: On-class
  - sign_mnist_train.csv:
    - ![eval_original](./pics/sign_mnist_original_eval.png)
  - sign_mnist_train_custom.csv:
    - ![eval_original](./pics/custom_eval.png)

## To-do
- test other models (P.Name)
- Web app: (Dat + H.Name)
    - front-end ?
    - back-end ?
        - web server (run test.py file to generate base64 code for image)
        - CPU or GPU spec?
- report