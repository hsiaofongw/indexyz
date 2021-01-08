import argparse
import os
from local import start_on_local_mode

class ServerModeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print('以服务器模式启动，端口号为：' + values[0])

class FolderModeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        fullPath = os.path.abspath(values[0])
        print('以目录模式启动，目录为：' + fullPath)
        
class JsonModeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        filePath = values[0]
        print('以JSON模式启动，JSON路径为：' + filePath)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--port',
    nargs=1, 
    metavar='port',
    help='work in server mode, need specify the port number.',
    dest='serverPort',
    action=ServerModeAction
)

parser.add_argument(
    '--folder',
    nargs=1,
    metavar='folder',
    help='select that folder that contain articles',
    dest='articlesFolder',
    action=FolderModeAction
)

parser.add_argument(
    '--json',
    nargs=1,
    metavar='json',
    help='select that json that contain articles',
    dest='jsonPath',
    action=ServerModeAction
)

args = parser.parse_args()