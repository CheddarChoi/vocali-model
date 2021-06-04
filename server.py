from flask import Flask, request, jsonify
from flask_script import Manager, Server
from flask_cors import CORS, cross_origin

from pydantic import BaseModel
from typing import List, Optional

import model, os

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

class VocaliServer(Flask):
  def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
    if not self.debug or os.getenv('WERKZEUG_RUN_MAIN') == 'true':
      with self.app_context():
        server_start()
    super(VocaliServer, self).run(host=host, port=port, debug=debug, load_dotenv=load_dotenv, **options)


port = int(os.environ.get("PORT", 5000))
app = VocaliServer(__name__)
CORS(app, resources={r'*': {'origins': '*'}})

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
    app.run(host="0.0.0.0", port=port)