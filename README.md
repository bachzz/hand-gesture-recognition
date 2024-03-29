# hand-gesture-recognition
Translating hand gestures to English letters

https://sign-language-app.herokuapp.com/


## Roles
- Bach: Data capture, deep learning model build, web deploy
- Dat: Data capture, build website, Random Forest classifier 
- H.Nam: Website building and deployment

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
  - Test dataset: On-class - sign_mnist_test_custom.csv
  - sign_mnist_train.csv:
    - ![eval_original](./pics/sign_mnist_original_eval.png)
  - sign_mnist_train_custom.csv:
    - ![eval_original](./pics/custom_eval.png)

## To-do
- Automatic detect hand gesture (localization)
- Classify complex gestures (time series data) or hand animations using LSTM, GRU 


### Run locally

IF YOU HAVE HEROKU:
- `heroku local`
IF NOT:
- `gunicorn -k eventlet -w 1 app:app --log-file=-`

- in your browser, navigate to localhost:5000

### Deploy to heroku

- `git push heroku master`
- heroku open

### Common Issues

If you run into a 'protocol not found' error, see if [this stackoverflow answer helps](https://stackoverflow.com/questions/40184788/protocol-not-found-socket-getprotobyname).
