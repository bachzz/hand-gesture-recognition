# hand-gesture-recognition
Translating hand gestures to English letters

## Roles
- Bach: generate data + preprocess data + model
- Dat: generate data + web app 
- Nam: generate data + web app

## To-do
- Generate data: https://drive.google.com/file/d/1G9HsXqpyc7MHf9C9lvWrM1LK20iHj94K/view
    - `python main.py`
    - change output path in code + press S to save data
    - G -> Y
    - 4 lightning schemes
    - vary distances: near + med + far
    - ~ 425 additional pics, each character
- Web app:
    - front-end ?
    - back-end ?
        - web server (run test.py file to generate base64 code for image)
        - CPU or GPU spec?

#### Optional

- setup heroku (`brew install heroku`)
- Use a python virtualenv

#### Required
- `git clone https://github.com/dxue2012/python-webcam-flask.git`
- `pip install -r requirements.txt`

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
