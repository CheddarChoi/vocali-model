from flask import Flask, request, jsonify
from flask_script import Manager, Server
from flask_cors import CORS, cross_origin

from pydantic import BaseModel
from typing import List, Optional

import model

class textField(BaseModel) :
  text: str

class UserInfo(BaseModel):
  prefWeight: Optional[float] = 0.5
  moodWeight: Optional[float] = 0.5
  pitchWeight: Optional[float] = 0.5
  likeList: Optional[List[str]] = []
  dislikeList: Optional[List[str]] = []
  undefinedList: Optional[List[str]] = []
  minPitch: Optional[str] = ''
  maxPitch: Optional[str] = ''
  moods: Optional[List[str]] = []

def server_start():
    model.init_model()

class CustomServer(Server):
    def __call__(self, app, *args, **kwargs):
        server_start()
        return Server.__call__(self, app, *args, **kwargs)

app = Flask(__name__)
CORS(app, resources={r'*': {'origins': '*'}})
manager = Manager(app)

# Remeber to add the command to your Manager instance
manager.add_command('runserver', CustomServer())

@app.route('/recommendations', methods = ['POST'])
def index():
    userInfo = request.get_json()
    result = model.send_output(
        [userInfo['prefWeight'], userInfo['moodWeight'], userInfo['pitchWeight']],
        userInfo['likeList'],
        userInfo['dislikeList'],
        userInfo['undefinedList'],
        userInfo['minPitch'],
        userInfo['maxPitch'],
        userInfo['moods']
    )
    print(result)
    return jsonify(result.to_dict('records')[:10])

@app.route('/', methods = ['POST'])
def test():
    return "hi!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    manager.run(host="0.0.0.0", port=port)
    # app.run()