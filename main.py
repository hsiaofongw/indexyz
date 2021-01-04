import argparse
import os
from local import start_on_local_mode

parser = argparse.ArgumentParser()

parser.add_argument(
    '--port',
    nargs=1, 
    metavar='port',
    help='work in server mode, need specify the port number.'
)

parser.add_argument(
    '--local', 
    nargs=1, 
    metavar='folder',
    help='work in local mode, need specify the data folder.'
)

args = parser.parse_args()

if args.port:
    port_num = int(args.port[0])
    print('现在以服务器模式启动．')
    print('监听的本地端口为：' + str(port_num))

if args.local:
    data_folder = args.local[0]
    print('现在以本地模式启动．')

    data_folder = args.local[0]
    print('数据文件夹为：' + os.path.abspath(data_folder))

    start_on_local_mode(data_folder)