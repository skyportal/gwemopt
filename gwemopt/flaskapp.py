import datetime
import os

from flask import Flask
from flask_humanize import Humanize
from werkzeug.routing import BaseConverter

# Application object
app = Flask(__name__, instance_relative_config=True)
app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql:///gwemopt'

app.config['TEMPLATES_AUTO_RELOAD'] = True

# Turn off memory-intensive modification tracking.
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Default secret key: secure and random. However, sessions are not preserved
# across different Python processes.
app.config['SECRET_KEY'] = os.urandom(24)

# Celery configuration.
# Use pickle serializer, because it supports byte values.
# Use redis broker, because it supports locks (and thus singleton tasks).
app.config['CELERY_BROKER_URL'] = 'redis://'
app.config['CELERY_ACCEPT_CONTENT'] = ['json', 'pickle']
app.config['CELERY_EVENT_SERIALIZER'] = 'json'
app.config['CELERY_RESULT_SERIALIZER'] = 'pickle'
app.config['CELERY_TASK_SERIALIZER'] = 'pickle'

# Apply instance config.
#app.config.from_pyfile('application.cfg', silent=False)


class DateTimeConverter(BaseConverter):

    def to_python(self, value):
        try:
            return datetime.datetime.strptime(value, '%y%m%d').date()
        except ValueError:
            return datetime.datetime.strptime(value, '%Y-%m-%dT%H:%M:%S')

    def to_url(self, value):
        return value.isoformat(timespec='seconds')


app.url_map.converters['datetime'] = DateTimeConverter

app.jinja_env.globals['now'] = datetime.datetime.now

humanize = Humanize(app)
