import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workers.tasks import celery_app
